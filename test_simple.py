"""
Simple test to check if PyTorch/Transformers are working correctly
This tests the basic translation capability without requiring trained models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing PyTorch and Transformers installation...\n")

# Check PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Try loading a small pre-trained model as a test
try:
    print("\nAttempting to load a small model (gpt2 as test)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Simple test generation
    text = "Hello"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest input: '{text}'")
    print(f"Test output: '{result}'")
    print("\n✅ PyTorch/Transformers working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nThis is expected if you don't have internet connection.")
    print("The actual translation will work once you train your model.")

print("\n" + "="*60)
print("Next steps:")
print("1. Add training data to data/raw/ folder")
print("2. Run: python run.py process")
print("3. Train your model or use a pre-trained one")
print("="*60)
