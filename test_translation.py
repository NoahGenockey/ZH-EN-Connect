"""
Simple translation test
"""

import torch
from transformers import MarianMTModel, MarianTokenizer

print("Loading model...")
model = MarianMTModel.from_pretrained('models/student/final_model')
tokenizer = MarianTokenizer.from_pretrained('models/student/final_model')
model.eval()

print("Model loaded!\n")

# Test translations
test_sentences = [
    "Hello, world!",
    "How are you today?",
    "This is a test of the translation system.",
    "The weather is nice today."
]

print("Running translations:\n")
for text in test_sentences:
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Translate
    with torch.no_grad():
        translated = model.generate(**inputs)
    
    # Decode
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    print(f"EN: {text}")
    print(f"ZH: {result}")
    print()

print("✅ Translation test complete!")
