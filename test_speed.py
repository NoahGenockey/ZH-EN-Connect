"""
Speed test for LinguaBridge translation optimizations.
Tests GPU availability and translation performance.
"""

import time
import torch
from src.utils import load_config
from src.inference import TranslationInference

def main():
    print("="*60)
    print("LinguaBridge - Performance Test")
    print("="*60)
    
    # Check PyTorch and CUDA
    print("\n1. System Information:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("   Device: CPU only")
    
    # Load model
    print("\n2. Loading Translation Model...")
    start_time = time.time()
    config = load_config()
    engine = TranslationInference(config)
    load_time = time.time() - start_time
    print(f"   Model loaded in {load_time:.2f}s")
    print(f"   Using device: {engine.device}")
    
    # Test single translation
    print("\n3. EN→ZH Translation Test:")
    test_sentence = "The implementation of artificial intelligence in healthcare diagnostics has revolutionized medical practice."
    
    # Warm-up
    _ = engine.translate(test_sentence, direction='en-zh')
    
    # Measure speed
    start_time = time.time()
    translation = engine.translate(test_sentence, direction='en-zh')
    elapsed = time.time() - start_time
    
    word_count = len(test_sentence.split())
    words_per_sec = word_count / elapsed
    
    print(f"   Input ({word_count} words): {test_sentence}")
    print(f"   Output: {translation}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Speed: {words_per_sec:.1f} words/sec")
    
    # Test ZH→EN translation
    print("\n4. ZH→EN Translation Test:")
    test_chinese = "人工智能在医疗诊断中的应用彻底改变了医疗实践。"
    
    start_time = time.time()
    translation_en = engine.translate(test_chinese, direction='zh-en')
    elapsed = time.time() - start_time
    
    print(f"   Input: {test_chinese}")
    print(f"   Output: {translation_en}")
    print(f"   Time: {elapsed:.3f}s")
    
    # Test batch translation
    print("\n5. Batch Translation Test:")
    test_batch = [
        "Hello, how are you today?",
        "The weather is beautiful outside.",
        "Machine learning is transforming technology.",
        "I enjoy reading books in my free time.",
        "Artificial intelligence continues to evolve rapidly."
    ]
    
    # Single processing (old method)
    start_time = time.time()
    single_results = [engine.translate(text, direction='en-zh') for text in test_batch]
    single_time = time.time() - start_time
    
    # Batch processing (new method)
    start_time = time.time()
    batch_results = engine.batch_translate(test_batch, batch_size=8, direction='en-zh')
    batch_time = time.time() - start_time
    
    speedup = single_time / batch_time
    
    print(f"   Single processing: {single_time:.3f}s")
    print(f"   Batch processing: {batch_time:.3f}s")
    print(f"   Speedup: {speedup:.2f}x faster")
    
    # Display results
    print("\n6. Batch Translation Results:")
    for i, (original, translated) in enumerate(zip(test_batch, batch_results), 1):
        print(f"   {i}. {original}")
        print(f"      → {translated}")
    
    # Performance summary
    print("\n" + "="*60)
    print("Performance Summary:")
    print("="*60)
    print(f"Device: {engine.device.upper()}")
    print(f"Translation Speed: {words_per_sec:.1f} words/sec")
    print(f"Batch Speedup: {speedup:.2f}x")
    print(f"Bidirectional: EN↔ZH supported ✅")
    
    if torch.cuda.is_available():
        print("\n✅ GPU acceleration is ENABLED")
        print("   Expected document translation time:")
        print("   - 100-page book: 8-15 minutes (vs 30-45 min CPU)")
        print("   - 300-page book: 25-45 minutes (vs 1.5-2 hours CPU)")
    else:
        print("\n⚠️  GPU acceleration is NOT available")
        print("   Running on CPU - translation will be slower")
        print("   To enable GPU:")
        print("   1. Install NVIDIA drivers")
        print("   2. Install CUDA Toolkit")
        print("   3. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
