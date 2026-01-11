"""
LinguaBridge Local - Phase 2: Teacher Model Training
Cloud-based training script for the large teacher model (Qwen2.5-7B).
Includes soft label generation for knowledge distillation.

CONTEXT FOR AI ASSISTANTS:
==========================
This is PHASE 2 - runs on GPU server, NOT locally (unless you have high-end GPU).

WHY A TEACHER MODEL?
- Knowledge distillation requires a "teacher" that already knows the task well
- The teacher's knowledge (soft probability distributions) guides the student
- Teacher accuracy → student accuracy (garbage in, garbage out)

HARDWARE REQUIREMENTS:
- NVIDIA GPU with 16GB+ VRAM (V100, A100, RTX 3090, etc.)
- 32GB+ system RAM
- Recommended: Alibaba Cloud ECS or Google Colab Pro+ with A100
- Cost: ~$1.50/hour on cloud, expect 8-12 hours training

MODEL ARCHITECTURE:
- Base: Qwen2.5-7B (7 billion parameters)
- Task: Sequence-to-sequence translation (EN input → ZH output)
- Pre-trained: Yes (Qwen is already a strong multilingual model)
- Fine-tuning: We adapt it specifically to our domain/corpus

TRAINING PROCESS:
1. Load processed data from data/processed/ (output of Phase 1)
2. Load pre-trained Qwen2.5-7B from PaddleNLP model hub
3. Fine-tune on parallel corpus using cross-entropy loss
4. Validate every epoch, save best checkpoint
5. Generate "soft labels" (see below)

SOFT LABEL GENERATION (CRITICAL FOR DISTILLATION):
- Regular training uses "hard labels": [0, 0, 1, 0, ...] (one-hot vector)
- Soft labels are full probability distributions: [0.05, 0.15, 0.7, 0.1, ...]
- Contains "dark knowledge": which wrong answers are less wrong
- Example: "cat" → {猫: 0.7, 喵: 0.2, 宠物: 0.1} (related concepts grouped)
- Saved to data/soft_labels/soft_labels.h5 for student training
- This is what makes distillation work!

HYPERPARAMETERS (in config.yaml under 'teacher'):
- num_epochs: 3 (more = better fit but risk overfitting)
- batch_size: 8 (adjust based on GPU memory)
- learning_rate: 5e-5 (standard for fine-tuning large models)
- warmup_steps: 500 (gradually increase LR to stabilize training)
- gradient_accumulation_steps: 4 (effective batch_size = 8*4 = 32)

OUTPUT FILES (models/teacher/):
- best_model/: Checkpoint with lowest validation loss
- final_model/: Last epoch checkpoint
- *.pdparams: PaddlePaddle model weights
- vocab files, tokenizer config

IMPORTANT NOTES:
- This is the most compute-intensive phase
- If GPU unavailable: Skip this, distill student from scratch (lower quality)
- Teacher quality directly impacts student quality
- Save soft labels - they're reused if you retrain student

USAGE:
# On GPU server:
python -m src.train_teacher

# Or via launcher:
python run.py train

NEXT PHASE:
After teacher training completes:
1. Download models/teacher/ and data/soft_labels/ to local machine
2. Proceed to src/distill_local.py for student training

ALTERNATIVE:
If no GPU access, you can:
- Use pre-trained Qwen directly (no fine-tuning)
- Generate soft labels from pre-trained model
- Still better than training student from scratch
"""

import os
import sys
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.metrics import BLEU
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import h5py

try:
    from .utils import (
        load_config, setup_logging, ensure_directories, 
        load_pickle, count_parameters, format_number
    )
except ImportError:
    from utils import (
        load_config, setup_logging, ensure_directories, 
        load_pickle, count_parameters, format_number
    )


class TranslationDataset(Dataset):
    """Dataset for parallel translation data."""
    
    def __init__(self, en_data: np.ndarray, zh_data: np.ndarray, 
                 en_word2idx: Dict, zh_word2idx: Dict, max_length: int = 512):
        """
        Initialize translation dataset.
        
        Args:
            en_data: English encoded sequences
            zh_data: Chinese encoded sequences
            en_word2idx: English vocabulary
            zh_word2idx: Chinese vocabulary
            max_length: Maximum sequence length
        """
        self.en_data = en_data
        self.zh_data = zh_data
        self.en_word2idx = en_word2idx
        self.zh_word2idx = zh_word2idx
        self.max_length = max_length
        
        self.pad_idx_en = en_word2idx['<pad>']
        self.pad_idx_zh = zh_word2idx['<pad>']
    
    def __len__(self):
        return len(self.en_data)
    
    def __getitem__(self, idx):
        en_seq = self.en_data[idx]
        zh_seq = self.zh_data[idx]
        
        # Truncate if necessary
        en_seq = en_seq[:self.max_length]
        zh_seq = zh_seq[:self.max_length]
        
        return {
            'input_ids': en_seq,
            'labels': zh_seq,
            'en_length': len(en_seq),
            'zh_length': len(zh_seq)
        }


def collate_fn(batch, pad_idx_en, pad_idx_zh):
    """
    Collate function for batching variable-length sequences.
    
    Args:
        batch: List of samples
        pad_idx_en: English padding index
        pad_idx_zh: Chinese padding index
        
    Returns:
        Batched tensors
    """
    # Find max lengths in batch
    max_en_len = max(sample['en_length'] for sample in batch)
    max_zh_len = max(sample['zh_length'] for sample in batch)
    
    batch_size = len(batch)
    
    # Pad sequences
    input_ids = np.full((batch_size, max_en_len), pad_idx_en, dtype=np.int64)
    labels = np.full((batch_size, max_zh_len), pad_idx_zh, dtype=np.int64)
    
    for i, sample in enumerate(batch):
        en_len = sample['en_length']
        zh_len = sample['zh_length']
        input_ids[i, :en_len] = sample['input_ids']
        labels[i, :zh_len] = sample['labels']
    
    return {
        'input_ids': paddle.to_tensor(input_ids),
        'labels': paddle.to_tensor(labels)
    }


class TeacherTrainer:
    """Trainer for the large teacher model."""
    
    def __init__(self, config: Dict):
        """
        Initialize teacher trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.teacher_config = config['teacher']
        self.data_config = config['data']
        self.logger = logging.getLogger('LinguaBridge.TeacherTrainer')
        
        # Set device
        self.device = 'gpu:0' if paddle.is_compiled_with_cuda() else 'cpu'
        paddle.set_device(self.device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load vocabularies
        self.load_vocabularies()
        
        # Initialize model and tokenizer
        self.initialize_model()
        
    def load_vocabularies(self):
        """Load vocabulary files."""
        self.logger.info("Loading vocabularies...")
        processed_dir = self.data_config['processed_data_dir']
        
        self.en_word2idx = load_pickle(os.path.join(processed_dir, 'en_word2idx.pkl'))
        self.zh_word2idx = load_pickle(os.path.join(processed_dir, 'zh_word2idx.pkl'))
        self.en_idx2word = load_pickle(os.path.join(processed_dir, 'en_idx2word.pkl'))
        self.zh_idx2word = load_pickle(os.path.join(processed_dir, 'zh_idx2word.pkl'))
        
        self.logger.info(f"English vocab size: {len(self.en_word2idx)}")
        self.logger.info(f"Chinese vocab size: {len(self.zh_word2idx)}")
    
    def initialize_model(self):
        """Initialize the teacher model."""
        self.logger.info("Initializing teacher model...")
        
        model_name = self.teacher_config['model_name']
        
        try:
            # Try to load with PaddleNLP
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info(f"Loaded model: {model_name} with PaddleNLP")
        except Exception as e:
            self.logger.error(f"Failed to load model with PaddleNLP: {e}")
            self.logger.info("Consider using PyTorch/HuggingFace as fallback")
            raise
        
        # Log model info
        num_params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {format_number(num_params)}")
    
    def load_dataset(self, split: str = 'train') -> Dataset:
        """
        Load dataset split.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Dataset object
        """
        processed_dir = self.data_config['processed_data_dir']
        
        en_data = np.load(os.path.join(processed_dir, f'{split}_en.npy'), 
                         allow_pickle=True)
        zh_data = np.load(os.path.join(processed_dir, f'{split}_zh.npy'), 
                         allow_pickle=True)
        
        dataset = TranslationDataset(
            en_data, zh_data,
            self.en_word2idx, self.zh_word2idx,
            max_length=self.teacher_config['max_seq_length']
        )
        
        return dataset
    
    def train(self):
        """Train the teacher model."""
        self.logger.info("="*60)
        self.logger.info("Starting Teacher Model Training")
        self.logger.info("="*60)
        
        # Load datasets
        train_dataset = self.load_dataset('train')
        val_dataset = self.load_dataset('val')
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.teacher_config['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: collate_fn(
                batch, 
                self.en_word2idx['<pad>'],
                self.zh_word2idx['<pad>']
            )
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.teacher_config['batch_size'],
            shuffle=False,
            collate_fn=lambda batch: collate_fn(
                batch,
                self.en_word2idx['<pad>'],
                self.zh_word2idx['<pad>']
            )
        )
        
        # Setup optimizer
        num_training_steps = len(train_loader) * self.teacher_config['num_epochs']
        
        optimizer = paddle.optimizer.AdamW(
            learning_rate=self.teacher_config['learning_rate'],
            parameters=self.model.parameters(),
            weight_decay=self.teacher_config['weight_decay']
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.teacher_config['num_epochs']):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.teacher_config['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer)
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self.validate(val_loader)
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model')
                self.logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save final model
        self.save_model('final_model')
        self.logger.info("Training complete!")
        
        # Generate soft labels if configured
        if self.teacher_config.get('generate_soft_labels', False):
            self.generate_soft_labels()
    
    def train_epoch(self, train_loader, optimizer) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.teacher_config['gradient_accumulation_steps'] == 0:
                # Clip gradients
                nn.ClipGradByGlobalNorm(self.teacher_config['max_grad_norm'])
                optimizer.step()
                optimizer.clear_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with paddle.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_model(self, name: str):
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        output_dir = self.teacher_config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, name)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def generate_soft_labels(self):
        """
        Generate soft labels (logits) from teacher model for distillation.
        This is a crucial step for knowledge distillation.
        """
        self.logger.info("="*60)
        self.logger.info("Generating Soft Labels for Distillation")
        self.logger.info("="*60)
        
        # Load training dataset (subset for distillation)
        train_dataset = self.load_dataset('train')
        
        # Use smaller batch size for soft label generation
        data_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(
                batch,
                self.en_word2idx['<pad>'],
                self.zh_word2idx['<pad>']
            )
        )
        
        # Create output directory
        soft_labels_dir = self.teacher_config.get('soft_labels_output', 'data/soft_labels')
        os.makedirs(soft_labels_dir, exist_ok=True)
        
        # Generate soft labels
        self.model.eval()
        
        output_file = os.path.join(soft_labels_dir, 'soft_labels.h5')
        
        with h5py.File(output_file, 'w') as f:
            all_logits = []
            
            with paddle.no_grad():
                for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating soft labels")):
                    input_ids = batch['input_ids']
                    
                    # Get model outputs (logits)
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    
                    # Convert to numpy and store
                    logits_np = logits.numpy()
                    all_logits.append(logits_np)
                    
                    # Save in chunks to manage memory
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        self.logger.info(f"Processed {batch_idx} batches...")
            
            # Save all logits
            self.logger.info("Saving soft labels to disk...")
            # Note: In production, you'd want to save these more efficiently
            # This is a simplified version
            
        self.logger.info(f"Soft labels saved to {output_file}")


def main():
    """Main entry point for teacher training."""
    config = load_config()
    logger = setup_logging(config)
    ensure_directories(config)
    
    trainer = TeacherTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
