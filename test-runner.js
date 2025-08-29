#!/usr/bin/env node

/**
 * Simple test runner for FlashFit AI
 * This script runs basic integration tests without external dependencies
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

class TestRunner {
  constructor() {
    this.tests = [];
    this.results = {
      passed: 0,
      failed: 0,
      total: 0
    };
  }

  // Add a test case
  test(name, testFn) {
    this.tests.push({ name, testFn });
  }

  // Run all tests
  async run() {
    console.log('🚀 Starting FlashFit AI Test Suite\n');
    
    for (const test of this.tests) {
      try {
        console.log(`⏳ Running: ${test.name}`);
        await test.testFn();
        console.log(`✅ PASSED: ${test.name}`);
        this.results.passed++;
      } catch (error) {
        console.log(`❌ FAILED: ${test.name}`);
        console.log(`   Error: ${error.message}`);
        this.results.failed++;
      }
      this.results.total++;
      console.log('');
    }

    this.printSummary();
  }

  // Print test results summary
  printSummary() {
    console.log('📊 Test Results Summary');
    console.log('========================');
    console.log(`Total Tests: ${this.results.total}`);
    console.log(`Passed: ${this.results.passed}`);
    console.log(`Failed: ${this.results.failed}`);
    console.log(`Success Rate: ${((this.results.passed / this.results.total) * 100).toFixed(1)}%`);
    
    if (this.results.failed > 0) {
      console.log('\n❌ Some tests failed!');
      process.exit(1);
    } else {
      console.log('\n🎉 All tests passed!');
    }
  }

  // Helper method to make HTTP requests
  async makeRequest(options, data = null) {
    return new Promise((resolve, reject) => {
      const protocol = options.protocol === 'https:' ? https : http;
      
      const req = protocol.request(options, (res) => {
        let body = '';
        res.on('data', chunk => body += chunk);
        res.on('end', () => {
          try {
            const jsonBody = body ? JSON.parse(body) : {};
            resolve({ statusCode: res.statusCode, body: jsonBody, headers: res.headers });
          } catch (e) {
            resolve({ statusCode: res.statusCode, body: body, headers: res.headers });
          }
        });
      });

      req.on('error', reject);
      
      if (data) {
        req.write(typeof data === 'string' ? data : JSON.stringify(data));
      }
      
      req.end();
    });
  }

  // Helper method to check if server is running
  async checkServer(url, timeout = 5000) {
    return new Promise((resolve) => {
      const urlObj = new URL(url);
      const options = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname,
        method: 'GET',
        timeout: timeout
      };

      const protocol = urlObj.protocol === 'https:' ? https : http;
      const req = protocol.request(options, (res) => {
        resolve(true);
      });

      req.on('error', () => resolve(false));
      req.on('timeout', () => resolve(false));
      req.end();
    });
  }

  // Assert helper
  assert(condition, message) {
    if (!condition) {
      throw new Error(message || 'Assertion failed');
    }
  }

  // Assert equal helper
  assertEqual(actual, expected, message) {
    if (actual !== expected) {
      throw new Error(message || `Expected ${expected}, but got ${actual}`);
    }
  }
}

// Create test runner instance
const runner = new TestRunner();

// Test: Check if backend server is accessible
runner.test('Backend Server Health Check', async () => {
  const isRunning = await runner.checkServer('http://localhost:8000');
  if (!isRunning) {
    throw new Error('Backend server is not running on port 8000');
  }
});

// Test: Check if frontend server is accessible
runner.test('Frontend Server Health Check', async () => {
  const isRunning = await runner.checkServer('http://localhost:3000');
  if (!isRunning) {
    throw new Error('Frontend server is not running on port 3000');
  }
});

// Test: Backend health endpoint
runner.test('Backend Health Endpoint', async () => {
  try {
    const response = await runner.makeRequest({
      hostname: 'localhost',
      port: 8000,
      path: '/health',
      method: 'GET'
    });
    
    runner.assertEqual(response.statusCode, 200, 'Health endpoint should return 200');
    runner.assert(response.body.status === 'healthy', 'Health status should be healthy');
  } catch (error) {
    throw new Error(`Health endpoint test failed: ${error.message}`);
  }
});

// Test: Backend API documentation endpoint
runner.test('API Documentation Endpoint', async () => {
  try {
    const response = await runner.makeRequest({
      hostname: 'localhost',
      port: 8000,
      path: '/docs',
      method: 'GET'
    });
    
    runner.assert(response.statusCode === 200 || response.statusCode === 307, 
      'Docs endpoint should be accessible');
  } catch (error) {
    throw new Error(`API docs test failed: ${error.message}`);
  }
});

// Test: File structure validation
runner.test('Project Structure Validation', async () => {
  const requiredFiles = [
    'backend/main.py',
    'backend/requirements.txt',
    'backend/Dockerfile',
    'frontend/package.json',
    'frontend/Dockerfile',
    'docker-compose.yml',
    'docker-compose.dev.yml',
    'docker-compose.prod.yml',
    'Makefile',
    'README.md'
  ];

  for (const file of requiredFiles) {
    const filePath = path.join(__dirname, file);
    if (!fs.existsSync(filePath)) {
      throw new Error(`Required file missing: ${file}`);
    }
  }
});

// Test: Environment configuration
runner.test('Environment Configuration Check', async () => {
  const envExamplePath = path.join(__dirname, '.env.example');
  
  if (!fs.existsSync(envExamplePath)) {
    throw new Error('.env.example file is missing');
  }

  const envContent = fs.readFileSync(envExamplePath, 'utf8');
  const requiredVars = [
    'API_HOST',
    'API_PORT',
    'FRONTEND_PORT',
    'REDIS_HOST',
    'SECRET_KEY'
  ];

  for (const varName of requiredVars) {
    if (!envContent.includes(varName)) {
      throw new Error(`Required environment variable ${varName} not found in .env.example`);
    }
  }
});

// Test: Docker configuration validation
runner.test('Docker Configuration Validation', async () => {
  const dockerComposePath = path.join(__dirname, 'docker-compose.yml');
  
  if (!fs.existsSync(dockerComposePath)) {
    throw new Error('docker-compose.yml file is missing');
  }

  const composeContent = fs.readFileSync(dockerComposePath, 'utf8');
  const requiredServices = ['backend', 'frontend', 'redis'];

  for (const service of requiredServices) {
    if (!composeContent.includes(service + ':')) {
      throw new Error(`Required service ${service} not found in docker-compose.yml`);
    }
  }
});

// Test: Package.json validation
runner.test('Frontend Package.json Validation', async () => {
  const packageJsonPath = path.join(__dirname, 'frontend/package.json');
  
  if (!fs.existsSync(packageJsonPath)) {
    throw new Error('frontend/package.json file is missing');
  }

  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  const requiredDeps = ['react', 'typescript', 'vite'];

  for (const dep of requiredDeps) {
    if (!packageJson.dependencies?.[dep] && !packageJson.devDependencies?.[dep]) {
      throw new Error(`Required dependency ${dep} not found in package.json`);
    }
  }
});

// Test: Backend requirements validation
runner.test('Backend Requirements Validation', async () => {
  const requirementsPath = path.join(__dirname, 'backend/requirements.txt');
  
  if (!fs.existsSync(requirementsPath)) {
    throw new Error('backend/requirements.txt file is missing');
  }

  const requirements = fs.readFileSync(requirementsPath, 'utf8');
  const requiredPackages = ['fastapi', 'uvicorn', 'redis', 'transformers'];

  for (const pkg of requiredPackages) {
    if (!requirements.includes(pkg)) {
      throw new Error(`Required package ${pkg} not found in requirements.txt`);
    }
  }
});

// Run all tests
if (require.main === module) {
  runner.run().catch(console.error);
}

module.exports = TestRunner;