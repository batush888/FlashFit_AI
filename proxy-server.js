const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 8080;

// Proxy API requests to backend
app.use('/api', createProxyMiddleware({
  target: 'http://localhost:3001',
  changeOrigin: true,
  pathRewrite: {
    '^/api': '/api' // Keep the /api prefix
  },
  logLevel: 'debug', // Add logging to debug the issue
  onError: (err, req, res) => {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: 'Backend service unavailable' });
  },
  onProxyReq: (proxyReq, req, res) => {
    console.log(`[PROXY] ${req.method} ${req.url} -> ${proxyReq.path}`);
  }
}));

// Serve static files from frontend build (if exists)
const frontendBuildPath = path.join(__dirname, 'frontend', 'dist');
if (fs.existsSync(frontendBuildPath)) {
  app.use(express.static(frontendBuildPath));
  
  // Fallback to index.html for SPA routing
  app.get('*', (req, res) => {
    res.sendFile(path.join(frontendBuildPath, 'index.html'));
  });
} else {
  // If no build exists, proxy frontend requests to dev server
  app.use('/', createProxyMiddleware({
    target: 'http://localhost:3000',
    changeOrigin: true,
    ws: true, // Enable WebSocket proxying for HMR
    onError: (err, req, res) => {
      console.error('Frontend proxy error:', err.message);
      res.status(500).send('Frontend service unavailable');
    }
  }));
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    services: {
      frontend: 'http://localhost:3000',
      backend: 'http://localhost:3001/api'
    }
  });
});

app.listen(PORT, () => {
  console.log(`\n🚀 Unified server running on http://localhost:${PORT}`);
  console.log('📱 Frontend: Proxied from http://localhost:3000');
  console.log('🔧 Backend API: Proxied from http://localhost:3001/api');
  console.log('\n✅ Both services now accessible on the same port!');
});