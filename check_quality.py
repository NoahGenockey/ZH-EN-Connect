"""
Quality check for CCMatrix dataset
Analyzes a large sample to determine actual data quality
"""

import random

def check_quality(file_path, sample_size=10000):
    """Check quality of Chinese text in dataset"""
    
    print(f"Analyzing {sample_size:,} lines from {file_path}...")
    
    # Count total lines first (streaming)
    print("Counting total lines...")
    total_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
            if total_lines % 1000000 == 0:
                print(f"  {total_lines:,} lines counted...")
    
    print(f"Total lines in file: {total_lines:,}")
    
    # Sample lines (streaming to avoid memory issues)
    print(f"Sampling {sample_size:,} lines...")
    sample_lines = []
    sample_interval = max(1, total_lines // sample_size)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % sample_interval == 0 and len(sample_lines) < sample_size:
                sample_lines.append(line)
    
    # Analyze quality
    bad_lines = 0
    short_lines = 0
    empty_lines = 0
    good_lines = 0
    
    bad_examples = []
    
    for line in sample_lines:
        line = line.strip()
        
        if not line:
            empty_lines += 1
            continue
        
        if len(line) < 5:
            short_lines += 1
            continue
        
        # Check for parentheses format
        paren_count = line.count('（') + line.count('）')
        if len(line) > 0 and paren_count / len(line) > 0.2:
            bad_lines += 1
            if len(bad_examples) < 3:
                bad_examples.append(line[:100])
        else:
            good_lines += 1
    
    print("\n" + "="*60)
    print("QUALITY ANALYSIS RESULTS")
    print("="*60)
    print(f"Sample size: {sample_size:,} lines")
    print(f"\nGood quality: {good_lines:,} ({good_lines/sample_size*100:.1f}%)")
    print(f"Bad format (parentheses): {bad_lines:,} ({bad_lines/sample_size*100:.1f}%)")
    print(f"Too short: {short_lines:,} ({short_lines/sample_size*100:.1f}%)")
    print(f"Empty: {empty_lines:,} ({empty_lines/sample_size*100:.1f}%)")
    
    if bad_examples:
        print("\nExamples of bad formatting:")
        for i, example in enumerate(bad_examples, 1):
            print(f"{i}. {example}...")
    
    print("\n" + "="*60)
    print(f"Expected usable data: {good_lines/sample_size*total_lines:,.0f} lines")
    print(f"Expected rejected: {(bad_lines+short_lines+empty_lines)/sample_size*total_lines:,.0f} lines")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    file_path = "data/raw/CCMatrix.en-zh.zh" if len(sys.argv) < 2 else sys.argv[1]
    sample_size = 50000  # 50k sample for statistical significance
    
    check_quality(file_path, sample_size)
