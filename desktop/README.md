# LinguaBridge Local - Desktop Application

Transform your translation system into a native desktop application!

## 🖥️ Desktop App Features

- **Native Desktop Experience**: Runs like any other desktop app (VS Code, Slack, etc.)
- **No Browser Required**: Self-contained application window
- **Auto-starts API**: Python server launches automatically
- **Native Menus**: File, Edit, View, Help menus
- **System Tray Support**: Minimize to tray (optional)
- **Cross-Platform**: Windows, macOS, Linux
- **One-Click Launch**: Double-click to start
- **Auto-updates**: Built-in update mechanism

## 📦 Installation

### Prerequisites

1. **Node.js** (v16 or higher)
   - Download from: https://nodejs.org
   - Verify: `node --version`

2. **Python** (3.8+)
   - Already installed for your project

### Quick Setup

```bash
# 1. Navigate to desktop folder
cd desktop

# 2. Install dependencies
npm install

# 3. Launch the app
npm start
```

### Easy Launch (Windows)

Double-click: **`start_desktop.bat`**

This automatically:
1. Checks dependencies
2. Installs if needed
3. Launches the desktop app

## 🚀 Usage

### Development Mode

```bash
cd desktop
npm start
```

### Build Executable

Create a standalone installer:

```bash
# Windows installer
npm run build:win

# macOS DMG
npm run build:mac

# Linux AppImage
npm run build:linux
```

Output in `desktop/dist/`

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│   Electron Desktop Window       │
│   (Chromium + Node.js)          │
│                                 │
│   ┌─────────────────────────┐  │
│   │  Web Frontend           │  │
│   │  (HTML/CSS/JS)          │  │
│   └──────────┬──────────────┘  │
└──────────────┼──────────────────┘
               │ HTTP (localhost)
┌──────────────▼──────────────────┐
│   Python API Server             │
│   (FastAPI + PyTorch)           │
│   Started automatically         │
└─────────────────────────────────┘
```

## 📂 File Structure

```
desktop/
├── package.json          # Node.js dependencies
├── main.js              # Electron main process
├── preload.js           # Security bridge
└── assets/              # App icons

web/
├── index.html           # Web UI (loaded in Electron)
├── style.css
├── app.js
└── loading.html         # Splash screen
```

## ⚙️ Configuration

### `package.json` - Build Settings

```json
{
  "build": {
    "appId": "com.linguabridge.local",
    "productName": "LinguaBridge Local",
    "win": {
      "target": "nsis",
      "icon": "assets/icon.ico"
    }
  }
}
```

### Window Settings (`main.js`)

```javascript
new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: 'LinguaBridge Local'
})
```

## 🎨 Customization

### App Icon

Place your icons in `desktop/assets/`:
- Windows: `icon.ico` (256x256)
- macOS: `icon.icns` (512x512)
- Linux: `icon.png` (512x512)

### Splash Screen

Edit `web/loading.html` for custom loading screen.

### Menu Bar

Customize in `main.js` → `menuTemplate`.

## 🔧 Development Tips

### Enable DevTools

Press `Ctrl+Shift+I` (or `Cmd+Option+I` on Mac)

### Reload App

Press `Ctrl+R` (or `Cmd+R` on Mac)

### Check Python Server

Open DevTools → Console → Should see:
```
Python: Server started...
Python: Model loaded...
```

## 📦 Distribution

### Windows Installer (NSIS)

```bash
npm run build:win
```

Creates: `dist/LinguaBridge-Local-Setup-1.0.0.exe`

**Features:**
- Installation wizard
- Desktop shortcut
- Start menu entry
- Uninstaller
- Auto-update support

### macOS DMG

```bash
npm run build:mac
```

Creates: `dist/LinguaBridge-Local-1.0.0.dmg`

**Features:**
- Drag-to-Applications
- Code signed (if certificate provided)
- Notarized (if configured)

### Linux AppImage

```bash
npm run build:linux
```

Creates: `dist/LinguaBridge-Local-1.0.0.AppImage`

**Features:**
- Single file distribution
- No installation required
- Runs on all major distros

## 🆚 Desktop vs Web vs GUI

| Feature | Electron Desktop | Web Frontend | Tkinter GUI |
|---------|-----------------|--------------|-------------|
| **Look & Feel** | Modern web UI | Browser-based | Native OS |
| **Installation** | One-time install | None | None |
| **Launch** | Double-click icon | Start server | Run Python |
| **Updates** | Auto-update | Refresh page | Manual |
| **File Size** | ~150 MB | ~30 KB | ~50 KB |
| **Packaging** | ✅ Installer | ❌ No | ⚠️ PyInstaller |
| **Distribution** | ✅ Easy | ❌ N/A | ⚠️ Complex |
| **Professional** | ✅✅✅ | ✅✅ | ✅ |

## 🎯 Why Electron?

**Pros:**
- ✅ Modern, beautiful UI (web tech)
- ✅ Easy distribution (installers)
- ✅ Cross-platform consistency
- ✅ Auto-update capability
- ✅ Native desktop integration
- ✅ Familiar to users (VS Code style)

**Cons:**
- ❌ Larger file size (~150 MB)
- ❌ More memory usage (~200 MB)
- ⚠️ Requires Node.js for development

## 🐛 Troubleshooting

### "Node.js not found"
Install from: https://nodejs.org

### "Python server failed to start"
- Check Python is in PATH
- Verify `run.py` works: `python run.py api`

### "Model not loading"
- Check `models/` directory exists
- Verify model files are present

### Build errors
```bash
# Clean and rebuild
cd desktop
rm -rf node_modules dist
npm install
npm run build:win
```

## 📊 Performance

**Memory Usage:**
- Electron: ~200 MB
- Python/Model: ~800 MB
- Total: ~1 GB

**Startup Time:**
- Electron window: < 1 second
- Python server: ~5 seconds
- Model loading: ~3 seconds
- **Total: ~8 seconds**

**Translation Speed:**
- Same as web frontend (~300ms)

## 🔐 Security

- `contextIsolation: true` - Separates web content
- `nodeIntegration: false` - No Node.js in renderer
- `preload.js` - Safe IPC bridge
- Local-only - No external connections

## 🎊 Success!

You now have THREE ways to use LinguaBridge:

1. **🖥️ Desktop App** (Electron) - Professional, distributable
2. **🌐 Web UI** (Browser) - Lightweight, modern
3. **🪟 Native GUI** (Tkinter) - Simple, traditional

Choose what works best for your users! 🚀
