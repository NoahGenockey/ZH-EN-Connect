"""
Convert Moses format parallel corpus to simple text files
Moses format has two files: source.lang1 and target.lang2
This script converts them to en.txt and zh.txt
"""

import os
import sys

def convert_moses_to_txt(source_file, target_file, output_dir="data/raw"):
    """
    Convert Moses format files to en.txt and zh.txt
    
    Args:
        source_file: Path to English source file (e.g., corpus.en)
        target_file: Path to Chinese target file (e.g., corpus.zh)
        output_dir: Output directory (default: data/raw)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    en_output = os.path.join(output_dir, "en.txt")
    zh_output = os.path.join(output_dir, "zh.txt")
    
    print(f"Converting Moses files...")
    print(f"Source: {source_file}")
    print(f"Target: {target_file}")
    print(f"Output: {en_output} and {zh_output}")
    
    # Read and convert
    with open(source_file, 'r', encoding='utf-8') as f_src, \
         open(target_file, 'r', encoding='utf-8') as f_tgt, \
         open(en_output, 'w', encoding='utf-8') as f_en, \
         open(zh_output, 'w', encoding='utf-8') as f_zh:
        
        line_count = 0
        for en_line, zh_line in zip(f_src, f_tgt):
            en_line = en_line.strip()
            zh_line = zh_line.strip()
            
            # Skip empty lines
            if not en_line or not zh_line:
                continue
            
            f_en.write(en_line + '\n')
            f_zh.write(zh_line + '\n')
            line_count += 1
            
            if line_count % 100000 == 0:
                print(f"Processed {line_count:,} lines...")
    
    print(f"\n✅ Conversion complete!")
    print(f"Total lines: {line_count:,}")
    print(f"Output files:")
    print(f"  - {en_output}")
    print(f"  - {zh_output}")
    print(f"\nNext step: python run.py process")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_moses.py <english_file> <chinese_file>")
        print("\nExample:")
        print("  python convert_moses.py corpus.en corpus.zh")
        print("  python convert_moses.py OpenSubtitles.en-zh.en OpenSubtitles.en-zh.zh")
        sys.exit(1)
    
    source_file = sys.argv[1]
    target_file = sys.argv[2]
    
    if not os.path.exists(source_file):
        print(f"❌ Error: Source file not found: {source_file}")
        sys.exit(1)
    
    if not os.path.exists(target_file):
        print(f"❌ Error: Target file not found: {target_file}")
        sys.exit(1)
    
    convert_moses_to_txt(source_file, target_file)
