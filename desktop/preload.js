// Preload script for Electron
// Provides safe bridge between main and renderer processes

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer process
contextBridge.exposeInMainWorld('electron', {
    // Get app version
    getVersion: () => ipcRenderer.invoke('get-app-version'),
    
    // Open URL in external browser
    openExternal: (url) => ipcRenderer.invoke('open-external', url),
    
    // Listen for events from main process
    onClearText: (callback) => {
        ipcRenderer.on('clear-text', callback);
    }
});

// Indicate we're running in Electron
contextBridge.exposeInMainWorld('isElectron', true);
