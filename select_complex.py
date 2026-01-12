"""
Select the most grammatically complex sentences for better translation quality.
Focuses on complex sentence structures to improve model's ability to handle
sophisticated texts like books and research papers.
"""

import heapq
import re
from tqdm import tqdm

def calculate_complexity_score(en_sentence, zh_sentence):
    """
    Calculate grammatical complexity score for a sentence pair.
    Higher score = more complex/sophisticated grammar.
    """
    score = 0
    
    # English complexity metrics
    en_words = en_sentence.split()
    en_word_count = len(en_words)
    
    # 1. Sentence length (longer = more complex, but cap at 100 words)
    score += min(en_word_count * 2, 200)
    
    # 2. Average word length (sophisticated vocabulary)
    if en_words:
        avg_word_len = sum(len(w.strip('.,!?;:')) for w in en_words) / len(en_words)
        score += avg_word_len * 10
    
    # 3. Punctuation variety (indicates complex clauses)
    punctuation_types = len(set(c for c in en_sentence if c in '.,;:—–-'))
    score += punctuation_types * 15
    
    # 4. Comma count (subordinate clauses, lists)
    comma_count = en_sentence.count(',')
    score += comma_count * 10
    
    # 5. Semicolons and colons (advanced sentence structure)
    score += en_sentence.count(';') * 25
    score += en_sentence.count(':') * 20
    
    # 6. Parenthetical expressions
    score += en_sentence.count('(') * 15
    
    # 7. Quotation marks (dialogue, citations)
    score += (en_sentence.count('"') + en_sentence.count("'")) * 5
    
    # 8. Complex conjunctions and subordinators
    subordinators = ['although', 'because', 'since', 'unless', 'whereas', 'while',
                     'however', 'moreover', 'furthermore', 'nevertheless', 'therefore',
                     'consequently', 'meanwhile', 'otherwise']
    en_lower = en_sentence.lower()
    score += sum(15 for word in subordinators if f' {word} ' in f' {en_lower} ')
    
    # 9. Relative clauses
    relative_pronouns = ['which', 'that', 'who', 'whom', 'whose', 'where', 'when']
    score += sum(10 for word in relative_pronouns if f' {word} ' in f' {en_lower} ')
    
    # 10. Passive voice indicators
    passive_indicators = ['was ', 'were ', 'been ', 'being ']
    score += sum(8 for phrase in passive_indicators if phrase in en_lower)
    
    # 11. Unique word ratio (vocabulary diversity)
    if en_word_count > 5:
        unique_ratio = len(set(w.lower() for w in en_words)) / en_word_count
        score += unique_ratio * 20
    
    # Chinese complexity metrics
    zh_chars = len(zh_sentence.strip())
    
    # 12. Chinese sentence length
    score += min(zh_chars, 200)
    
    # 13. Chinese punctuation
    zh_punct_count = sum(zh_sentence.count(p) for p in '，。；：、？！""''（）')
    score += zh_punct_count * 8
    
    # 14. Penalize very short sentences (usually simple)
    if en_word_count < 5 or zh_chars < 10:
        score *= 0.3
    
    # 15. Penalize extremely long sentences (often formatting errors)
    if en_word_count > 100 or zh_chars > 300:
        score *= 0.5
    
    return score


def select_top_complex_sentences(en_file, zh_file, output_en, output_zh, top_n=2000000):
    """
    Select the top N most complex sentence pairs using a min-heap.
    Memory efficient for large datasets.
    """
    print(f"Selecting top {top_n:,} most complex sentences...")
    print("This may take 30-60 minutes for 71M sentences...\n")
    
    # Min-heap to track top N sentences: (score, index, en_line, zh_line)
    heap = []
    
    # Count total lines
    print("Counting lines...")
    with open(en_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total: {total_lines:,} sentence pairs\n")
    
    # Score all sentences
    print("Scoring sentences by grammatical complexity...")
    with open(en_file, 'r', encoding='utf-8') as f_en, \
         open(zh_file, 'r', encoding='utf-8') as f_zh:
        
        for idx, (en_line, zh_line) in enumerate(tqdm(zip(f_en, f_zh), total=total_lines)):
            en_line = en_line.strip()
            zh_line = zh_line.strip()
            
            if not en_line or not zh_line:
                continue
            
            score = calculate_complexity_score(en_line, zh_line)
            
            if len(heap) < top_n:
                # Heap not full, add item
                heapq.heappush(heap, (score, idx, en_line, zh_line))
            elif score > heap[0][0]:
                # Found a higher scoring sentence, replace lowest
                heapq.heapreplace(heap, (score, idx, en_line, zh_line))
    
    # Sort by original index to preserve some document coherence
    print(f"\nSorting selected {len(heap):,} sentences...")
    selected = sorted(heap, key=lambda x: x[1])
    
    # Write output
    print(f"Writing to {output_en} and {output_zh}...")
    with open(output_en, 'w', encoding='utf-8') as f_en, \
         open(output_zh, 'w', encoding='utf-8') as f_zh:
        
        for score, idx, en_line, zh_line in selected:
            f_en.write(en_line + '\n')
            f_zh.write(zh_line + '\n')
    
    # Statistics
    scores = [item[0] for item in selected]
    print(f"\n{'='*60}")
    print("✅ Selection Complete!")
    print('='*60)
    print(f"Selected: {len(selected):,} sentence pairs")
    print(f"Complexity score range: {min(scores):.1f} - {max(scores):.1f}")
    print(f"Average complexity: {sum(scores)/len(scores):.1f}")
    print(f"\nOutput files:")
    print(f"  - {output_en}")
    print(f"  - {output_zh}")


if __name__ == "__main__":
    # Input: full 71M dataset
    en_input = "data/raw/en.txt"
    zh_input = "data/raw/zh.txt"
    
    # Output: 2M most complex sentences
    en_output = "data/raw/en_complex_2M.txt"
    zh_output = "data/raw/zh_complex_2M.txt"
    
    select_top_complex_sentences(en_input, zh_input, en_output, zh_output, top_n=2000000)
