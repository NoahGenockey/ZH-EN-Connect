"""
Batch processor for large datasets
Automatically splits and processes data in chunks to avoid memory errors
"""

import os
import subprocess
import sys

def split_large_dataset(en_file, zh_file, chunk_size=5000000, output_dir="data/raw/chunks"):
    """Split large dataset into manageable chunks"""
    print(f"Splitting dataset into {chunk_size:,} line chunks...")
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_num = 0
    
    with open(en_file, 'r', encoding='utf-8') as f_en, \
         open(zh_file, 'r', encoding='utf-8') as f_zh:
        
        while True:
            # Read chunk
            en_lines = []
            zh_lines = []
            
            for _ in range(chunk_size):
                en_line = f_en.readline()
                zh_line = f_zh.readline()
                
                if not en_line or not zh_line:
                    break
                    
                en_lines.append(en_line)
                zh_lines.append(zh_line)
            
            if not en_lines:
                break
            
            # Write chunk
            chunk_en = os.path.join(output_dir, f"chunk_{chunk_num}_en.txt")
            chunk_zh = os.path.join(output_dir, f"chunk_{chunk_num}_zh.txt")
            
            with open(chunk_en, 'w', encoding='utf-8') as out_en:
                out_en.writelines(en_lines)
            with open(chunk_zh, 'w', encoding='utf-8') as out_zh:
                out_zh.writelines(zh_lines)
            
            print(f"Created chunk {chunk_num}: {len(en_lines):,} lines")
            chunk_num += 1
    
    print(f"\n✅ Created {chunk_num} chunks")
    return chunk_num

def process_chunks(num_chunks, chunk_dir="data/raw/chunks"):
    """Process each chunk automatically"""
    print(f"\nProcessing {num_chunks} chunks automatically...")
    
    for i in range(num_chunks):
        print(f"\n{'='*60}")
        print(f"Processing chunk {i+1}/{num_chunks}")
        print('='*60)
        
        # Copy chunk to main data location
        chunk_en = os.path.join(chunk_dir, f"chunk_{i}_en.txt")
        chunk_zh = os.path.join(chunk_dir, f"chunk_{i}_zh.txt")
        
        # Backup existing files
        if os.path.exists("data/raw/en.txt"):
            os.replace("data/raw/en.txt", "data/raw/en_backup.txt")
        if os.path.exists("data/raw/zh.txt"):
            os.replace("data/raw/zh.txt", "data/raw/zh_backup.txt")
        
        # Copy chunk
        import shutil
        shutil.copy(chunk_en, "data/raw/en.txt")
        shutil.copy(chunk_zh, "data/raw/zh.txt")
        
        # Process
        result = subprocess.run([sys.executable, "run.py", "process"], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error processing chunk {i}")
            print(result.stderr)
            continue
        
        # Move processed data to chunk-specific location
        chunk_output = f"data/processed/chunk_{i}"
        os.makedirs(chunk_output, exist_ok=True)
        
        for file in os.listdir("data/processed"):
            if file.endswith(('.npy', '.pkl')):
                src = os.path.join("data/processed", file)
                dst = os.path.join(chunk_output, file)
                shutil.move(src, dst)
        
        print(f"✅ Chunk {i+1} processed successfully")
    
    print(f"\n{'='*60}")
    print("All chunks processed!")
    print('='*60)

def combine_chunks(num_chunks):
    """Combine processed chunks into final dataset"""
    print("\nCombining all chunks into final dataset...")
    
    import numpy as np
    import pickle
    
    # Combine training data
    all_train_en = []
    all_train_zh = []
    all_val_en = []
    all_val_zh = []
    all_test_en = []
    all_test_zh = []
    
    for i in range(num_chunks):
        chunk_dir = f"data/processed/chunk_{i}"
        
        train_en = np.load(os.path.join(chunk_dir, "train_en.npy"), allow_pickle=True)
        train_zh = np.load(os.path.join(chunk_dir, "train_zh.npy"), allow_pickle=True)
        val_en = np.load(os.path.join(chunk_dir, "val_en.npy"), allow_pickle=True)
        val_zh = np.load(os.path.join(chunk_dir, "val_zh.npy"), allow_pickle=True)
        test_en = np.load(os.path.join(chunk_dir, "test_en.npy"), allow_pickle=True)
        test_zh = np.load(os.path.join(chunk_dir, "test_zh.npy"), allow_pickle=True)
        
        all_train_en.extend(train_en)
        all_train_zh.extend(train_zh)
        all_val_en.extend(val_en)
        all_val_zh.extend(val_zh)
        all_test_en.extend(test_en)
        all_test_zh.extend(test_zh)
        
        print(f"Added chunk {i}: {len(train_en):,} train samples")
    
    # Save combined data
    np.save("data/processed/train_en.npy", all_train_en)
    np.save("data/processed/train_zh.npy", all_train_zh)
    np.save("data/processed/val_en.npy", all_val_en)
    np.save("data/processed/val_zh.npy", all_val_zh)
    np.save("data/processed/test_en.npy", all_test_en)
    np.save("data/processed/test_zh.npy", all_test_zh)
    
    # Use vocab from last chunk (they should all be similar)
    last_chunk = f"data/processed/chunk_{num_chunks-1}"
    import shutil
    for vocab_file in ['en_word2idx.pkl', 'en_idx2word.pkl', 'zh_word2idx.pkl', 'zh_idx2word.pkl']:
        shutil.copy(os.path.join(last_chunk, vocab_file), f"data/processed/{vocab_file}")
    
    print(f"\n✅ Combined dataset created!")
    print(f"Total training samples: {len(all_train_en):,}")
    print(f"Total validation samples: {len(all_val_en):,}")
    print(f"Total test samples: {len(all_test_en):,}")

def main():
    print("="*60)
    print("Batch Data Processor for Large Datasets")
    print("="*60)
    
    en_file = "data/raw/en.txt"
    zh_file = "data/raw/zh.txt"
    chunk_size = 5000000  # 5M lines per chunk
    
    # Check if files exist
    if not os.path.exists(en_file) or not os.path.exists(zh_file):
        print("❌ Error: data/raw/en.txt or zh.txt not found")
        sys.exit(1)
    
    # Step 1: Split
    num_chunks = split_large_dataset(en_file, zh_file, chunk_size)
    
    # Step 2: Process each chunk
    process_chunks(num_chunks)
    
    # Step 3: Combine results
    combine_chunks(num_chunks)
    
    print("\n" + "="*60)
    print("✅ COMPLETE! Dataset ready for training.")
    print("="*60)

if __name__ == "__main__":
    main()
