"""
Convert Moses format with quality filtering
Skips lines with excessive parentheses or encoding issues
"""

import os
import sys
import re

def is_valid_chinese(text):
    """Check if Chinese text is valid (not the parentheses format)"""
    # Count parentheses
    paren_count = text.count('（') + text.count('）')
    # If more than 20% of characters are parentheses, it's probably bad
    if len(text) > 0 and paren_count / len(text) > 0.2:
        return False
    return True

def convert_moses_filtered(source_file, target_file, output_dir="data/raw"):
    """Convert Moses files with quality filtering"""
    os.makedirs(output_dir, exist_ok=True)
    
    en_output = os.path.join(output_dir, "en.txt")
    zh_output = os.path.join(output_dir, "zh.txt")
    
    print(f"Converting Moses files with quality filter...")
    print(f"Source: {source_file}")
    print(f"Target: {target_file}")
    
    line_count = 0
    skipped = 0
    
    with open(source_file, 'r', encoding='utf-8') as f_src, \
         open(target_file, 'r', encoding='utf-8') as f_tgt, \
         open(en_output, 'w', encoding='utf-8') as f_en, \
         open(zh_output, 'w', encoding='utf-8') as f_zh:
        
        for en_line, zh_line in zip(f_src, f_tgt):
            en_line = en_line.strip()
            zh_line = zh_line.strip()
            
            # Skip empty lines
            if not en_line or not zh_line:
                skipped += 1
                continue
            
            # Skip lines with bad Chinese formatting
            if not is_valid_chinese(zh_line):
                skipped += 1
                continue
            
            # Skip very short lines (likely low quality)
            if len(en_line) < 5 or len(zh_line) < 3:
                skipped += 1
                continue
            
            f_en.write(en_line + '\n')
            f_zh.write(zh_line + '\n')
            line_count += 1
            
            if line_count % 100000 == 0:
                print(f"Processed {line_count:,} lines (skipped {skipped:,})...")
    
    print(f"\n✅ Conversion complete!")
    print(f"Valid lines: {line_count:,}")
    print(f"Skipped lines: {skipped:,}")
    print(f"Quality rate: {line_count/(line_count+skipped)*100:.1f}%")
    print(f"\nOutput files:")
    print(f"  - {en_output}")
    print(f"  - {zh_output}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_moses_filtered.py <english_file> <chinese_file>")
        sys.exit(1)
    
    source_file = sys.argv[1]
    target_file = sys.argv[2]
    
    if not os.path.exists(source_file):
        print(f"❌ Error: Source file not found: {source_file}")
        sys.exit(1)
    
    if not os.path.exists(target_file):
        print(f"❌ Error: Target file not found: {target_file}")
        sys.exit(1)
    
    convert_moses_filtered(source_file, target_file)
