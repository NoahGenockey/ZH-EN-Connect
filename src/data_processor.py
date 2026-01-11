"""
LinguaBridge Local - Phase 1: Data Processing Pipeline
Handles cleaning, tokenization, vocabulary building, and dataset formatting.

CONTEXT FOR AI ASSISTANTS:
==========================
This is the FIRST phase of the pipeline. It must run before any training.

INPUT REQUIREMENTS:
- data/raw/en.txt: English sentences, one per line, UTF-8 encoded
- data/raw/zh.txt: Chinese sentences, one per line, UTF-8 encoded
- Line N in en.txt must correspond to line N in zh.txt (parallel corpus)
- Recommended: 10k-1M sentence pairs (more = better model quality)

WHAT THIS MODULE DOES:
1. CLEANING:
   - Removes malformed Unicode, normalizes whitespace
   - Filters sentences by length (5-100 words configurable in config.yaml)
   - Checks alignment (EN/ZH length ratio 0.5-2.0 to catch misaligned pairs)

2. TOKENIZATION:
   - English: Moses tokenizer (industry standard for machine translation)
     * Handles contractions, punctuation, etc.
   - Chinese: Jieba segmenter (best for Mandarin word segmentation)
     * Chinese doesn't have spaces between words, jieba adds them
   - Preserves sentence boundaries for seq2seq training

3. VOCABULARY BUILDING:
   - Creates word2idx (token→integer) and idx2word (integer→token) mappings
   - Default: Top 50k most frequent tokens per language
   - Includes special tokens: <pad>=0, <unk>=1, <s>=2 (BOS), </s>=3 (EOS)
   - Frequency filtering: min_freq=2 removes one-off typos/noise

4. ENCODING:
   - Converts tokenized sentences to integer sequences
   - Format: [<s>, token1_idx, token2_idx, ..., </s>]
   - Saved as NumPy arrays (dtype=object for variable-length sequences)

5. SPLITTING:
   - Train: 95% (for model training)
   - Validation: 3% (for hyperparameter tuning, early stopping)
   - Test: 2% (for final evaluation, never seen during training)

OUTPUT FILES (data/processed/):
- train_en.npy, train_zh.npy: Training set integer sequences
- val_en.npy, val_zh.npy: Validation set
- test_en.npy, test_zh.npy: Test set
- en_word2idx.pkl, en_idx2word.pkl: English vocabulary mappings
- zh_word2idx.pkl, zh_idx2word.pkl: Chinese vocabulary mappings

CRITICAL NOTES:
- Vocabularies MUST be saved - inference needs same word2idx to encode new text
- If you reprocess data, retrain models (vocab indices may change)
- Larger vocab = better coverage but more memory/compute
- Moses/Jieba must be installed: pip install sacremoses jieba

USAGE:
python -m src.data_processor
# Or via launcher:
python run.py process

NEXT PHASE:
After this completes, proceed to either:
- src/train_teacher.py (if you have GPU server access)
- src/distill_local.py (skip teacher, train student from scratch - lower quality)
"""

import os
import re
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import jieba
from sacremoses import MosesTokenizer, MosesDetokenizer
from pathlib import Path
import logging

try:
    from .utils import load_config, setup_logging, ensure_directories, save_pickle, ProgressTracker
except ImportError:
    from utils import load_config, setup_logging, ensure_directories, save_pickle, ProgressTracker


class DataProcessor:
    """
    Data processing pipeline for English-Chinese parallel corpus.
    Handles cleaning, tokenization, vocabulary building, and formatting.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataProcessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.logger = logging.getLogger('LinguaBridge.DataProcessor')
        
        # Initialize tokenizers
        self.en_tokenizer = MosesTokenizer(lang='en')
        self.en_detokenizer = MosesDetokenizer(lang='en')
        
        # Chinese tokenizer (jieba)
        jieba.setLogLevel(logging.INFO)
        
        # Vocabulary dictionaries
        self.en_vocab = {}
        self.zh_vocab = {}
        self.en_vocab_inv = {}
        self.zh_vocab_inv = {}
        
    def clean_text(self, text: str, language: str = 'en') -> str:
        """
        Clean raw text: normalize whitespace, remove malformed Unicode.
        
        Args:
            text: Raw text string
            language: 'en' or 'zh'
            
        Returns:
            Cleaned text
        """
        # Remove malformed Unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text
    
    def tokenize_english(self, text: str) -> List[str]:
        """
        Tokenize English text using Moses tokenizer.
        
        Args:
            text: English text
            
        Returns:
            List of tokens
        """
        return self.en_tokenizer.tokenize(text, return_str=False)
    
    def tokenize_chinese(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using jieba.
        
        Args:
            text: Chinese text
            
        Returns:
            List of tokens
        """
        return list(jieba.cut(text))
    
    def is_valid_sentence_pair(self, en_tokens: List[str], zh_tokens: List[str]) -> bool:
        """
        Check if sentence pair meets filtering criteria.
        
        Args:
            en_tokens: English tokens
            zh_tokens: Chinese tokens
            
        Returns:
            True if valid, False otherwise
        """
        en_len = len(en_tokens)
        zh_len = len(zh_tokens)
        
        min_len = self.data_config['min_word_count']
        max_len = self.data_config['max_word_count']
        
        # Check length constraints
        if en_len < min_len or en_len > max_len:
            return False
        if zh_len < min_len or zh_len > max_len:
            return False
        
        # Check length ratio (avoid misaligned pairs)
        ratio = en_len / max(zh_len, 1)
        if ratio < 0.5 or ratio > 2.0:
            return False
        
        return True
    
    def load_and_clean_data(self) -> Tuple[List[str], List[str]]:
        """
        Load and clean raw parallel text files.
        
        Returns:
            Tuple of (English sentences, Chinese sentences)
        """
        self.logger.info("Loading raw data files...")
        
        en_file = os.path.join(self.data_config['raw_data_dir'], self.data_config['en_file'])
        zh_file = os.path.join(self.data_config['raw_data_dir'], self.data_config['zh_file'])
        
        if not os.path.exists(en_file) or not os.path.exists(zh_file):
            raise FileNotFoundError(
                f"Raw data files not found. Please place {self.data_config['en_file']} "
                f"and {self.data_config['zh_file']} in {self.data_config['raw_data_dir']}"
            )
        
        # Read files
        with open(en_file, 'r', encoding='utf-8') as f:
            en_lines = f.readlines()
        
        with open(zh_file, 'r', encoding='utf-8') as f:
            zh_lines = f.readlines()
        
        if len(en_lines) != len(zh_lines):
            raise ValueError(f"Mismatch: {len(en_lines)} English lines vs {len(zh_lines)} Chinese lines")
        
        self.logger.info(f"Loaded {len(en_lines)} sentence pairs")
        
        # Clean data
        self.logger.info("Cleaning text...")
        en_cleaned = []
        zh_cleaned = []
        
        tracker = ProgressTracker(len(en_lines), "Cleaning")
        for en_line, zh_line in zip(en_lines, zh_lines):
            en_clean = self.clean_text(en_line, 'en')
            zh_clean = self.clean_text(zh_line, 'zh')
            
            if en_clean and zh_clean:  # Skip empty lines
                en_cleaned.append(en_clean)
                zh_cleaned.append(zh_clean)
            
            tracker.update()
        tracker.finish()
        
        self.logger.info(f"Cleaned data: {len(en_cleaned)} valid sentence pairs")
        return en_cleaned, zh_cleaned
    
    def tokenize_and_filter(self, en_sentences: List[str], 
                           zh_sentences: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Tokenize sentences and filter based on length criteria.
        
        Args:
            en_sentences: List of English sentences
            zh_sentences: List of Chinese sentences
            
        Returns:
            Tuple of (English token lists, Chinese token lists)
        """
        self.logger.info("Tokenizing and filtering sentences...")
        
        en_tokenized = []
        zh_tokenized = []
        
        tracker = ProgressTracker(len(en_sentences), "Tokenizing")
        for en_sent, zh_sent in zip(en_sentences, zh_sentences):
            en_tokens = self.tokenize_english(en_sent)
            zh_tokens = self.tokenize_chinese(zh_sent)
            
            if self.is_valid_sentence_pair(en_tokens, zh_tokens):
                en_tokenized.append(en_tokens)
                zh_tokenized.append(zh_tokens)
            
            tracker.update()
        tracker.finish()
        
        self.logger.info(f"After filtering: {len(en_tokenized)} valid sentence pairs")
        return en_tokenized, zh_tokenized
    
    def build_vocabulary(self, tokenized_sentences: List[List[str]], 
                        language: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Build vocabulary from tokenized sentences.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            language: 'en' or 'zh'
            
        Returns:
            Tuple of (word2idx, idx2word) dictionaries
        """
        self.logger.info(f"Building {language} vocabulary...")
        
        # Count token frequencies
        token_counter = Counter()
        for tokens in tokenized_sentences:
            token_counter.update(tokens)
        
        self.logger.info(f"Total unique tokens: {len(token_counter)}")
        
        # Select top tokens by frequency
        vocab_size = self.data_config['vocab_size']
        min_freq = self.data_config['min_freq']
        
        # Filter by minimum frequency
        filtered_tokens = [token for token, freq in token_counter.items() if freq >= min_freq]
        
        # Sort by frequency and take top vocab_size
        most_common = token_counter.most_common(vocab_size)
        selected_tokens = [token for token, _ in most_common]
        
        # Build word2idx with special tokens
        special_tokens = self.data_config['special_tokens']
        word2idx = {
            special_tokens['pad']: 0,
            special_tokens['unk']: 1,
            special_tokens['bos']: 2,
            special_tokens['eos']: 3,
        }
        
        for token in selected_tokens:
            if token not in word2idx:
                word2idx[token] = len(word2idx)
        
        # Build idx2word
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        self.logger.info(f"Final {language} vocabulary size: {len(word2idx)}")
        return word2idx, idx2word
    
    def encode_sentences(self, tokenized_sentences: List[List[str]], 
                        word2idx: Dict[str, int]) -> List[List[int]]:
        """
        Convert tokenized sentences to integer sequences.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            word2idx: Word to index mapping
            
        Returns:
            List of integer sequences
        """
        special_tokens = self.data_config['special_tokens']
        unk_idx = word2idx[special_tokens['unk']]
        bos_idx = word2idx[special_tokens['bos']]
        eos_idx = word2idx[special_tokens['eos']]
        
        encoded = []
        for tokens in tokenized_sentences:
            indices = [bos_idx]
            indices.extend([word2idx.get(token, unk_idx) for token in tokens])
            indices.append(eos_idx)
            encoded.append(indices)
        
        return encoded
    
    def split_dataset(self, en_encoded: List[List[int]], 
                     zh_encoded: List[List[int]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            en_encoded: Encoded English sequences
            zh_encoded: Encoded Chinese sequences
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        self.logger.info("Splitting dataset...")
        
        total_size = len(en_encoded)
        train_ratio = self.data_config['train_split']
        val_ratio = self.data_config['val_split']
        
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(total_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        def gather_split(split_indices):
            en_split = [en_encoded[i] for i in split_indices]
            zh_split = [zh_encoded[i] for i in split_indices]
            return en_split, zh_split
        
        splits = {
            'train': gather_split(train_indices),
            'val': gather_split(val_indices),
            'test': gather_split(test_indices)
        }
        
        self.logger.info(f"Split sizes - Train: {len(splits['train'][0])}, "
                        f"Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}")
        
        return splits
    
    def save_processed_data(self, splits: Dict, en_vocab: Tuple, zh_vocab: Tuple) -> None:
        """
        Save processed data and vocabularies to disk.
        
        Args:
            splits: Dataset splits
            en_vocab: (word2idx, idx2word) for English
            zh_vocab: (word2idx, idx2word) for Chinese
        """
        self.logger.info("Saving processed data...")
        
        output_dir = self.data_config['processed_data_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits as numpy arrays
        for split_name, (en_data, zh_data) in splits.items():
            # Save as list of variable-length sequences
            np.save(os.path.join(output_dir, f'{split_name}_en.npy'), 
                   np.array(en_data, dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f'{split_name}_zh.npy'), 
                   np.array(zh_data, dtype=object), allow_pickle=True)
        
        # Save vocabularies
        en_word2idx, en_idx2word = en_vocab
        zh_word2idx, zh_idx2word = zh_vocab
        
        save_pickle(en_word2idx, os.path.join(output_dir, 'en_word2idx.pkl'))
        save_pickle(en_idx2word, os.path.join(output_dir, 'en_idx2word.pkl'))
        save_pickle(zh_word2idx, os.path.join(output_dir, 'zh_word2idx.pkl'))
        save_pickle(zh_idx2word, os.path.join(output_dir, 'zh_idx2word.pkl'))
        
        self.logger.info(f"Processed data saved to {output_dir}")
    
    def process(self) -> None:
        """
        Run the complete data processing pipeline.
        """
        self.logger.info("="*60)
        self.logger.info("Starting Data Processing Pipeline")
        self.logger.info("="*60)
        
        # Step 1: Load and clean
        en_sentences, zh_sentences = self.load_and_clean_data()
        
        # Step 2: Tokenize and filter
        en_tokenized, zh_tokenized = self.tokenize_and_filter(en_sentences, zh_sentences)
        
        # Step 3: Build vocabularies
        en_word2idx, en_idx2word = self.build_vocabulary(en_tokenized, 'en')
        zh_word2idx, zh_idx2word = self.build_vocabulary(zh_tokenized, 'zh')
        
        self.en_vocab = (en_word2idx, en_idx2word)
        self.zh_vocab = (zh_word2idx, zh_idx2word)
        
        # Step 4: Encode sentences
        self.logger.info("Encoding sentences...")
        en_encoded = self.encode_sentences(en_tokenized, en_word2idx)
        zh_encoded = self.encode_sentences(zh_tokenized, zh_word2idx)
        
        # Step 5: Split dataset
        splits = self.split_dataset(en_encoded, zh_encoded)
        
        # Step 6: Save everything
        self.save_processed_data(splits, self.en_vocab, self.zh_vocab)
        
        self.logger.info("="*60)
        self.logger.info("Data Processing Complete!")
        self.logger.info("="*60)


def main():
    """Main entry point for data processing."""
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logging(config)
    
    # Ensure directories exist
    ensure_directories(config)
    
    # Create processor and run pipeline
    processor = DataProcessor(config)
    processor.process()


if __name__ == "__main__":
    main()
