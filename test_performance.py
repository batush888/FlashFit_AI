#!/usr/bin/env python3
"""
FlashFit AI Performance Testing Suite
Tests UX flow, API latency, and system performance
"""

import asyncio
import aiohttp
import time
import json
import os
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PerformanceResult:
    endpoint: str
    method: str
    avg_latency: float
    min_latency: float
    max_latency: float
    success_rate: float
    total_requests: int

class FlashFitPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.test_results: List[PerformanceResult] = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def authenticate(self, email: str = "test@performance.com", password: str = "testpass123"):
        """Authenticate and get token for protected endpoints"""
        # First register if user doesn't exist
        register_data = {
            "email": email,
            "password": password,
            "full_name": "Performance Test User"
        }
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        try:
            async with self.session.post(f"{self.base_url}/api/auth/register", json=register_data) as resp:
                if resp.status in [200, 201]:
                    print("✓ Test user registered successfully")
        except Exception as e:
            print(f"Registration failed (user may already exist): {e}")
        
        # Login to get token
        login_data = {"email": email, "password": password}
        async with self.session.post(f"{self.base_url}/api/auth/login", json=login_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                # Handle the nested response structure
                if "data" in result and "token" in result["data"]:
                    self.auth_token = result["data"]["token"]
                else:
                    self.auth_token = result.get("access_token")
                
                if self.auth_token:
                    self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
                    print("✓ Authentication successful")
                    return True
                else:
                    print("✗ No token received in response")
                    return False
            else:
                print(f"✗ Authentication failed: {resp.status}")
                text = await resp.text()
                print(f"Response: {text}")
                return False
    
    async def benchmark_endpoint(self, endpoint: str, method: str = "GET", 
                               data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, str]] = None, 
                               num_requests: int = 10) -> PerformanceResult:
        """Benchmark a specific endpoint"""
        print(f"\n🔄 Benchmarking {method} {endpoint} ({num_requests} requests)...")
        
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        latencies = []
        successes = 0
        
        for i in range(num_requests):
            start_time = time.time()
            status = 500  # Default error status
            
            try:
                if method.upper() == "GET":
                    async with self.session.get(f"{self.base_url}{endpoint}") as resp:
                        status = resp.status
                elif method.upper() == "POST":
                    if files:
                        # Handle file upload
                        form_data = aiohttp.FormData()
                        for key, file_path in files.items():
                            if os.path.exists(file_path):
                                form_data.add_field(key, open(file_path, 'rb'), filename=os.path.basename(file_path))
                        async with self.session.post(f"{self.base_url}{endpoint}", data=form_data) as resp:
                            status = resp.status
                    else:
                        async with self.session.post(f"{self.base_url}{endpoint}", json=data) as resp:
                            status = resp.status
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
                
                if 200 <= status < 300:
                    successes += 1
                    
                print(f"  Request {i+1}/{num_requests}: {latency:.2f}ms (Status: {status})")
                
            except Exception as e:
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                latencies.append(latency)
                print(f"  Request {i+1}/{num_requests}: ERROR - {e}")
        
        if latencies:
            result = PerformanceResult(
                endpoint=endpoint,
                method=method,
                avg_latency=statistics.mean(latencies),
                min_latency=min(latencies),
                max_latency=max(latencies),
                success_rate=(successes / num_requests) * 100,
                total_requests=num_requests
            )
        else:
            result = PerformanceResult(
                endpoint=endpoint,
                method=method,
                avg_latency=0,
                min_latency=0,
                max_latency=0,
                success_rate=0,
                total_requests=num_requests
            )
        
        self.test_results.append(result)
        return result
    
    async def test_ux_flow(self):
        """Test complete UX flow: registration → login → upload → wardrobe → recommendations"""
        print("\n🎯 Testing Complete UX Flow...")
        
        # 1. Test Authentication Flow
        print("\n1️⃣ Testing Authentication...")
        auth_success = await self.authenticate()
        if not auth_success:
            print("❌ Authentication failed - cannot continue UX flow test")
            return False
        
        # 2. Test Profile Access
        print("\n2️⃣ Testing Profile Access...")
        await self.benchmark_endpoint("/api/user/profile", "GET", num_requests=3)
        
        # 3. Test File Upload (if test image exists)
        print("\n3️⃣ Testing File Upload...")
        test_image_path = "test_image.jpg"
        if not os.path.exists(test_image_path):
            # Create a simple test image file
            print("Creating test image for upload testing...")
            try:
                from PIL import Image
                import numpy as np
                
                # Create a simple 100x100 RGB image
                img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(test_image_path, "JPEG")
                print("✓ Test image created")
            except ImportError:
                print("⚠️ PIL not available, skipping upload test")
                test_image_path = None
        
        if test_image_path and os.path.exists(test_image_path):
            await self.benchmark_endpoint("/api/upload", "POST", files={"file": test_image_path}, num_requests=3)
        
        # 4. Test Wardrobe Access
        print("\n4️⃣ Testing Wardrobe Access...")
        await self.benchmark_endpoint("/api/wardrobe", "GET", num_requests=5)
        
        # 5. Test Match Generation (Critical Performance Test)
        print("\n5️⃣ Testing Match Generation (Target: <3s)...")
        match_data = {
            "occasion": "casual",
            "weather": "mild",
            "style_preference": "modern"
        }
        result = await self.benchmark_endpoint("/api/match", "POST", data=match_data, num_requests=5)
        
        # Check if match generation meets performance target
        if result.avg_latency > 3000:  # 3 seconds in milliseconds
            print(f"⚠️ WARNING: Match generation average latency ({result.avg_latency:.2f}ms) exceeds 3s target")
        else:
            print(f"✅ Match generation performance meets target: {result.avg_latency:.2f}ms < 3000ms")
        
        return True
    
    async def stress_test_uploads(self, num_concurrent: int = 5, num_files: int = 10):
        """Stress test file uploads"""
        print(f"\n💪 Stress Testing Uploads ({num_concurrent} concurrent, {num_files} files each)...")
        
        # Create test files if they don't exist
        test_files = []
        for i in range(num_files):
            filename = f"stress_test_{i}.jpg"
            if not os.path.exists(filename):
                try:
                    from PIL import Image
                    import numpy as np
                    
                    # Create random images of different sizes
                    size = 50 + (i * 10)  # Varying sizes from 50x50 to 140x140
                    img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    img.save(filename, "JPEG")
                    test_files.append(filename)
                except ImportError:
                    print("⚠️ PIL not available, skipping stress test")
                    return
            else:
                test_files.append(filename)
        
        # Run concurrent uploads
        tasks = []
        for i in range(num_concurrent):
            for filename in test_files:
                task = self.benchmark_endpoint("/api/upload", "POST", files={"file": filename}, num_requests=1)
                tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_uploads = num_concurrent * num_files
        
        print(f"\n📊 Stress Test Results:")
        print(f"   Total uploads: {total_uploads}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Throughput: {total_uploads/total_time:.2f} uploads/second")
        
        # Cleanup test files
        for filename in test_files:
            try:
                os.remove(filename)
            except:
                pass
    
    def generate_report(self):
        """Generate performance test report"""
        print("\n" + "="*60)
        print("📋 PERFORMANCE TEST REPORT")
        print("="*60)
        
        if not self.test_results:
            print("No test results available.")
            return
        
        print(f"{'Endpoint':<25} {'Method':<8} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Success %':<10}")
        print("-" * 80)
        
        for result in self.test_results:
            print(f"{result.endpoint:<25} {result.method:<8} {result.avg_latency:<10.2f} "
                  f"{result.min_latency:<10.2f} {result.max_latency:<10.2f} {result.success_rate:<10.1f}")
        
        # Performance Analysis
        print("\n🔍 PERFORMANCE ANALYSIS:")
        
        # Check critical endpoints
        match_results = [r for r in self.test_results if "/api/match" in r.endpoint]
        if match_results:
            avg_match_latency = statistics.mean([r.avg_latency for r in match_results])
            if avg_match_latency > 3000:
                print(f"❌ CRITICAL: Match generation too slow ({avg_match_latency:.2f}ms > 3000ms target)")
            else:
                print(f"✅ Match generation performance acceptable ({avg_match_latency:.2f}ms)")
        
        # Check overall success rates
        avg_success_rate = statistics.mean([r.success_rate for r in self.test_results])
        if avg_success_rate < 95:
            print(f"⚠️ WARNING: Low overall success rate ({avg_success_rate:.1f}%)")
        else:
            print(f"✅ Good overall success rate ({avg_success_rate:.1f}%)")
        
        # Save detailed report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [
                {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "avg_latency_ms": r.avg_latency,
                    "min_latency_ms": r.min_latency,
                    "max_latency_ms": r.max_latency,
                    "success_rate_percent": r.success_rate,
                    "total_requests": r.total_requests
                }
                for r in self.test_results
            ]
        }
        
        with open("performance_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: performance_report.json")

async def main():
    """Main testing function"""
    print("🚀 FlashFit AI Performance Testing Suite")
    print("========================================\n")
    
    async with FlashFitPerformanceTester() as tester:
        # Test complete UX flow
        await tester.test_ux_flow()
        
        # Additional endpoint benchmarks
        print("\n🎯 Additional Endpoint Benchmarks...")
        
        # Test available endpoints (some may not exist)
        try:
            await tester.benchmark_endpoint("/health", "GET", num_requests=5)
        except Exception as e:
            print(f"Health endpoint test failed: {e}")
        
        try:
            await tester.benchmark_endpoint("/api/history/statistics", "GET", num_requests=3)
        except Exception as e:
            print(f"History statistics test failed: {e}")
        
        # Stress test uploads
        await tester.stress_test_uploads(num_concurrent=3, num_files=5)
        
        # Generate final report
        tester.generate_report()
        
        print("\n✅ Performance testing completed!")
        print("\n📝 RECOMMENDATIONS:")
        print("   1. Monitor match generation latency closely (target: <3s)")
        print("   2. Consider implementing caching for frequent wardrobe queries")
        print("   3. Optimize image processing pipeline for faster uploads")
        print("   4. Set up continuous performance monitoring in production")
        print("   5. Consider ML model optimization (ONNX/quantization) if latency issues persist")

if __name__ == "__main__":
    asyncio.run(main())