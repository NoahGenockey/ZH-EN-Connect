# LinguaBridge Local - Web Frontend

Modern, responsive web interface for English-to-Chinese translation.

## 🌟 Features

- **Modern UI**: Clean, professional design with gradient backgrounds
- **Real-time Translation**: Fast translation with visual feedback
- **Smart Caching**: Instant results for repeated translations
- **Statistics Dashboard**: Track translations and cache performance
- **Keyboard Shortcuts**: Power-user friendly (Ctrl+Enter to translate)
- **Download & Copy**: Easy export of translations
- **100% Offline**: All processing happens locally
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🚀 Quick Start

### 1. Start the API Server

```bash
python run.py api
```

The server will start at `http://localhost:8000`

### 2. Open Web Interface

Open your browser and navigate to:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Files

- `index.html` - Main HTML structure
- `style.css` - Modern CSS styling with CSS variables
- `app.js` - Frontend logic and API communication

## ⌨️ Keyboard Shortcuts

- `Ctrl + Enter` - Translate text
- `Ctrl + C` - Copy translation (when focused on output)
- `Ctrl + K` - Clear all text

## 🎨 UI Components

### Translation Panel
- **Source Input**: Multi-line English text input with character counter
- **Translation Button**: Large, prominent action button
- **Target Output**: Read-only Chinese translation display
- **Action Buttons**: Clear, Paste, Copy, Download

### Info Panel
- **Statistics**: Translation count, cache hits, API status
- **Tips**: Best practices for using the translator
- **Shortcuts**: Quick reference for keyboard shortcuts

## 🔧 Configuration

The frontend automatically connects to the API at `http://localhost:8000`. To change this, edit `app.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## 🌐 API Endpoints Used

- `GET /health` - Check API status
- `POST /translate` - Translate text
- `GET /cache/stats` - Get cache statistics

## 📊 Features Comparison

| Feature | Tkinter GUI | Web Frontend |
|---------|-------------|--------------|
| Installation | None (built-in) | None (pure HTML/CSS/JS) |
| User Experience | Native desktop | Modern web app |
| Multi-user | No | Yes (via network) |
| Mobile Support | No | Yes |
| Remote Access | No | Yes |
| File Size | ~50KB | ~30KB |

## 🎯 Best Practices

1. **Short Texts**: Best for sentences up to 512 words
2. **Complex Sentences**: Handles sophisticated grammar well
3. **Batch Translation**: Use multiple calls for large documents
4. **Cache**: Repeated translations are instant

## 🐛 Troubleshooting

### "API offline" Error
- Ensure the API server is running: `python run.py api`
- Check http://localhost:8000/health in browser

### Translation Not Working
- Check browser console (F12) for errors
- Verify API is loaded (status indicator should show "Online")

### CORS Errors
- API includes CORS middleware for local development
- All origins are allowed by default

## 🔮 Future Enhancements

- [ ] Dark mode toggle
- [ ] Translation history
- [ ] Document upload and translation
- [ ] Multiple language pairs
- [ ] Real-time translation as you type
- [ ] Export to various formats (PDF, DOCX)
- [ ] Pronunciation guide (Pinyin)

## 💡 Technology Stack

- **Frontend**: Pure HTML5, CSS3, JavaScript (ES6+)
- **Backend**: FastAPI (Python)
- **Model**: Helsinki-NLP/opus-mt-en-zh (Marian NMT)
- **No Build Tools**: Works directly in browser

## 📝 License

Same as the main project - see LICENSE file.
