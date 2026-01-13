# Bidirectional Translation - EN↔ZH

## ✅ What's New

LinguaBridge now supports **bidirectional translation**! You can translate:
- **English → Chinese** (EN→ZH)
- **Chinese → English** (ZH→EN)

## 🎯 Features

### Both Translation Directions:
- ✅ English → Chinese (EN→ZH)
- ✅ Chinese → English (ZH→EN)
- ✅ Same speed optimizations apply to both
- ✅ GPU acceleration for both models
- ✅ Batch processing for documents

### All Interfaces Updated:
- ✅ **Enhanced GUI** (LinguaBridge-Enhanced.bat) - Direction selector in both tabs
- ✅ **Simple GUI** (LinguaBridge.bat) - Direction selector added
- ✅ **Text Translation** - Radio buttons to choose EN→ZH or ZH→EN
- ✅ **Document Translation** - Translate PDFs/EPUBs in either direction

---

## 🚀 How to Use

### Text Translation:
1. Launch: `LinguaBridge-Enhanced.bat` or `LinguaBridge.bat`
2. Select direction: **English → Chinese** or **Chinese → English**
3. Enter text and click Translate
4. Labels automatically update based on direction!

### Document Translation:
1. Open **LinguaBridge-Enhanced.bat**
2. Go to "📚 Document Translation" tab
3. Select direction: EN→ZH or ZH→EN
4. Choose your PDF or EPUB file
5. Translate entire books in either direction!

### Quick Test:
```bash
python test_speed.py
```

This will test both directions:
- English → Chinese
- Chinese → English

---

## 🎯 What Changed:

### ✅ Bidirectional Translation
- **EN → ZH**: English to Chinese (existing model: opus-mt-en-zh)
- **ZH → EN**: Chinese to English (new model: Helsinki-NLP opus-mt-zh-en, 312MB)

### ✅ Updated Components:
1. **[inference.py](src/inference.py)** - Loads both models, direction parameter
2. **[app_gui.py](src/app_gui.py)** - Simple GUI with direction selector
3. **[app_gui_enhanced.py](src/app_gui_enhanced.py)** - Enhanced GUI with direction selector
4. **[document_translator.py](src/document_translator.py)** - Supports both directions
5. **[config.yaml](config.yaml)** - Added ZH→EN model path
6. **[test_speed.py](test_speed.py)** - Tests both directions

### 🎨 GUI Changes:
- Added direction selector (radio buttons) in both GUIs
- Input/output labels automatically update based on selection
- "English Input" ↔ "Chinese Input"
- "Chinese Translation" ↔ "English Translation"

### 📚 Document Translation:
- PDF translation now supports both directions
- EPUB translation now supports both directions
- Same batch processing for maximum speed

---

## 📊 Models Used

| Direction | Model | Size | Quality |
|-----------|-------|------|---------|
| EN → ZH | Helsinki-NLP/opus-mt-en-zh | ~300MB | BLEU 31.4 |
| ZH → EN | Helsinki-NLP/opus-mt-zh-en | ~312MB | BLEU 28.5 |

Both models:
- MarianMT architecture (fast, efficient)
- Trained on millions of sentence pairs
- Production-quality translations
- GPU-accelerated (when available)

---

## 🧪 Testing

### Test Both Directions:
```bash
python test_speed.py
```

Output shows:
- GPU availability
- EN→ZH translation speed
- ZH→EN translation speed
- Batch processing speedup

### Example Translations:

**EN → ZH:**
```
Input:  The implementation of artificial intelligence in healthcare has revolutionized medical practice.
Output: 在医疗诊断中实施人工智能彻底改变了医疗实践。
```

**ZH → EN:**
```
Input:  人工智能在医疗诊断中的应用彻底改变了医疗实践。
Output: The application of artificial intelligence in medical diagnosis has revolutionized medical practice.
```

---

## 💡 Tips

1. **For best quality**: Keep default beam_size=4
2. **For faster speed**: Set beam_size=2 in config.yaml
3. **GPU recommended**: 3-5x faster for both directions
4. **Batch processing**: Automatically used for documents
5. **Cache works**: Translations cached per direction

---

## 🚧 First-Time Setup

When you first use ZH→EN translation:
- The system will download the ZH→EN model (~312MB)
- This happens automatically on first use
- Download progress shown in console
- Model cached locally for future use
- Only needs to download once!

---

## 🎉 Summary

You can now translate **both ways**:
- ✅ English → Chinese
- ✅ Chinese → English
- ✅ Text translation (instant)
- ✅ Document translation (PDF/EPUB)
- ✅ All speed optimizations apply
- ✅ GPU acceleration for both
- ✅ Simple GUI interface

**Just select your direction and translate!** 🌉
