"""
Quick test to verify PyTorch is working without downloading models
"""

import torch
from transformers import __version__ as transformers_version

print("="*60)
print("PyTorch/Transformers Installation Test")
print("="*60)

# Test PyTorch
print(f"\n✅ PyTorch version: {torch.__version__}")
print(f"✅ Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Test basic tensor operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y
print(f"✅ Basic tensor ops work: {x.tolist()} + {y.tolist()} = {z.tolist()}")

# Test transformers
print(f"✅ Transformers version: {transformers_version}")

print("\n" + "="*60)
print("SUCCESS! All core dependencies are working correctly.")
print("="*60)
print("\nNext steps:")
print("1. Your code has been updated to use PyTorch")
print("2. The inference engine (src/inference.py) is ready")
print("3. You can now:")
print("   - Add training data to data/raw/")
print("   - Or use pre-trained models from Hugging Face")
print("   - Run: python run.py gui (for desktop app)")
print("   - Run: python run.py api (for web API)")
print("="*60)
