// LinguaBridge Local - Electron Main Process
// This file controls the desktop application lifecycle

const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;
const API_PORT = 8000;

// Create the desktop window
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 800,
        minHeight: 600,
        title: 'LinguaBridge Local - EN→ZH Translation',
        icon: path.join(__dirname, '../assets/icon.png'),
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        backgroundColor: '#667eea',
        show: false // Don't show until ready
    });

    // Create application menu
    const menuTemplate = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'New Translation',
                    accelerator: 'CmdOrCtrl+N',
                    click: () => mainWindow.webContents.send('clear-text')
                },
                { type: 'separator' },
                {
                    label: 'Exit',
                    accelerator: 'CmdOrCtrl+Q',
                    click: () => app.quit()
                }
            ]
        },
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectAll' }
            ]
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: 'Help',
            submenu: [
                {
                    label: 'About',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'About LinguaBridge Local',
                            message: 'LinguaBridge Local v1.0.0',
                            detail: 'Offline English-to-Chinese Neural Machine Translation\n\n' +
                                   'Powered by Helsinki-NLP/opus-mt-en-zh\n' +
                                   '78M parameters, trained on 20-50M sentence pairs\n\n' +
                                   '100% offline • Privacy-first • Fast & accurate',
                            buttons: ['OK']
                        });
                    }
                },
                {
                    label: 'API Documentation',
                    click: () => {
                        require('electron').shell.openExternal('http://localhost:8000/docs');
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(menuTemplate);
    Menu.setApplicationMenu(menu);

    // Show splash screen while loading
    mainWindow.loadFile(path.join(__dirname, '../web/loading.html'));
    mainWindow.show();

    // Start Python API server
    startPythonServer();

    // Wait for server to be ready, then load main UI
    waitForServer().then(() => {
        mainWindow.loadURL('http://localhost:8000');
    }).catch(err => {
        showErrorDialog('Failed to start translation engine', err.message);
    });

    // Handle window close
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// Start Python API server
function startPythonServer() {
    const isDev = !app.isPackaged;
    const projectRoot = isDev ? path.join(__dirname, '..') : process.resourcesPath;
    
    console.log('Starting Python server...');
    console.log('Project root:', projectRoot);

    // Determine Python executable
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    
    pythonProcess = spawn(pythonCmd, ['run.py', 'api'], {
        cwd: projectRoot,
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
    });
}

// Wait for server to be ready
function waitForServer(maxAttempts = 30) {
    return new Promise((resolve, reject) => {
        let attempts = 0;
        
        const checkServer = () => {
            attempts++;
            
            fetch(`http://localhost:${API_PORT}/health`)
                .then(res => res.json())
                .then(data => {
                    if (data.model_loaded) {
                        console.log('Server is ready!');
                        resolve();
                    } else if (attempts < maxAttempts) {
                        setTimeout(checkServer, 1000);
                    } else {
                        reject(new Error('Server started but model not loaded'));
                    }
                })
                .catch(err => {
                    if (attempts < maxAttempts) {
                        setTimeout(checkServer, 1000);
                    } else {
                        reject(new Error('Server failed to start'));
                    }
                });
        };
        
        // Start checking after 2 seconds
        setTimeout(checkServer, 2000);
    });
}

// Show error dialog
function showErrorDialog(title, message) {
    dialog.showErrorBox(title, message);
    app.quit();
}

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

// Cleanup on quit
app.on('before-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

app.on('will-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

// IPC Handlers
ipcMain.handle('get-app-version', () => {
    return app.getVersion();
});

ipcMain.handle('open-external', (event, url) => {
    require('electron').shell.openExternal(url);
});
