const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('config', {
  API_BASE_URL: process.env.API_BASE_URL || 'http://127.0.0.1:8000'
});
