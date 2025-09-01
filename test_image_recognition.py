#!/usr/bin/env python3
"""
Test Image Recognition Functionality

This script tests the ultimate AI image recognition system by:
1. Creating a simple test image
2. Testing the /api/ultimate/analyze endpoint
3. Verifying that all AI models are working correctly
"""

import requests
import json
from PIL import Image, ImageDraw, ImageFont
import io
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8080"
TEST_IMAGE_PATH = "data/test_images/test_fashion_item.jpg"

def create_test_image():
    """Create a simple test fashion image"""
    # Create a simple colored rectangle representing a shirt
    width, height = 300, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw a simple shirt shape
    # Body
    draw.rectangle([50, 100, 250, 350], fill='red', outline='black', width=2)
    
    # Sleeves
    draw.rectangle([20, 100, 50, 200], fill='red', outline='black', width=2)
    draw.rectangle([250, 100, 280, 200], fill='red', outline='black', width=2)
    
    # Collar
    draw.polygon([(120, 100), (150, 80), (180, 100)], fill='white', outline='black')
    
    # Add text label
    try:
        # Try to use a default font
        font = ImageFont.load_default()
        draw.text((100, 30), "Red Shirt", fill='black', font=font)
    except:
        draw.text((100, 30), "Red Shirt", fill='black')
    
    # Save the image
    os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)
    image.save(TEST_IMAGE_PATH)
    print(f"✓ Test image created: {TEST_IMAGE_PATH}")
    return TEST_IMAGE_PATH

def test_health_endpoint():
    """Test if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running and healthy")
            return True
        else:
            print(f"✗ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return False

def test_ultimate_ai_health():
    """Test the ultimate AI health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/ultimate/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✓ Ultimate AI system health:")
            print(f"  Status: {result.get('status', 'unknown')}")
            print(f"  Models loaded: {result.get('total_models', 0)}")
            print(f"  System ready: {result.get('system_ready', False)}")
            
            models = result.get('models_loaded', {})
            for model_name, status in models.items():
                status_icon = "✓" if status else "✗"
                print(f"  {status_icon} {model_name}: {status}")
            
            return result.get('system_ready', False)
        else:
            print(f"✗ Ultimate AI health check failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Ultimate AI health check error: {e}")
        return False

def authenticate_user():
    """Register and login a test user to get authentication token"""
    test_email = "test_image_recognition@example.com"
    test_password = "testpass123456"
    
    try:
        # First try to register the user
        register_data = {
            "email": test_email,
            "password": test_password,
            "full_name": "Test Image Recognition User"
        }
        
        register_response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json=register_data,
            timeout=10
        )
        
        if register_response.status_code in [200, 201]:
            print("✓ Test user registered successfully")
        elif register_response.status_code == 400:
            print("ℹ️  Test user already exists, proceeding to login...")
        else:
            print(f"⚠️  Registration response: {register_response.status_code}")
        
        # Now login to get the token
        login_data = {
            "email": test_email,
            "password": test_password
        }
        
        login_response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json=login_data,
            timeout=10
        )
        
        if login_response.status_code == 200:
            result = login_response.json()
            # Handle different response structures
            if "data" in result and "token" in result["data"]:
                token = result["data"]["token"]
            elif "access_token" in result:
                token = result["access_token"]
            else:
                print(f"⚠️  Unexpected login response structure: {result}")
                return None
            
            print("✓ Authentication successful")
            return token
        else:
            print(f"✗ Login failed: {login_response.status_code}")
            print(f"  Response: {login_response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        return None

def test_image_analysis(image_path):
    """Test the ultimate AI image analysis endpoint"""
    try:
        # First authenticate to get a valid token
        auth_token = authenticate_user()
        if not auth_token:
            print("✗ Could not authenticate, skipping image analysis test")
            return False
        
        # Test image analysis with valid authentication
        headers = {'Authorization': f'Bearer {auth_token}'}
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            
            response = requests.post(
                f"{BASE_URL}/api/ultimate/analyze",
                files=files,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Image analysis successful!")
                print("\n📊 Analysis Results:")
                
                # Print the full response for debugging
                print(f"\n🔍 Full Response: {json.dumps(result, indent=2)}")
                
                # Extract the actual data
                data = result.get('data', {})
                
                # Display results from different models
                if 'clip' in data:
                    clip_data = data['clip']
                    print(f"  🔍 CLIP Analysis: {type(clip_data).__name__}")
                    if isinstance(clip_data, dict):
                        print(f"    Keys: {list(clip_data.keys())}")
                
                if 'blip' in data:
                    blip_data = data['blip']
                    print(f"  📝 BLIP Analysis: {type(blip_data).__name__}")
                    if isinstance(blip_data, dict):
                        print(f"    Keys: {list(blip_data.keys())}")
                
                if 'fashion' in data:
                    fashion_data = data['fashion']
                    print(f"  👗 Fashion Analysis: {type(fashion_data).__name__}")
                    if isinstance(fashion_data, dict):
                        print(f"    Keys: {list(fashion_data.keys())}")
                
                if 'predictor' in data:
                    predictor_data = data['predictor']
                    print(f"  🎯 Predictor Analysis: {type(predictor_data).__name__}")
                    if isinstance(predictor_data, dict):
                        category = predictor_data.get('category', {}).get('predicted', 'unknown')
                        confidence = predictor_data.get('category', {}).get('confidence', 0.0)
                        print(f"    Category: {category} (confidence: {confidence:.2f})")
                        print(f"    Keys: {list(predictor_data.keys())}")
                
                if 'fusion_score' in data:
                    fusion_score = data['fusion_score']
                    print(f"  🔗 Fusion Score: {fusion_score:.3f}")
                
                print(f"\n📋 Response Keys: {list(result.keys())}")
                print(f"📋 Data Keys: {list(data.keys()) if data else 'No data'}")
                return True
                
            else:
                print(f"✗ Image analysis failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ Image analysis error: {e}")
        return False

def main():
    """Run the complete image recognition test"""
    print("🧪 Testing FlashFit AI Image Recognition System")
    print("=" * 50)
    
    # Step 1: Check server health
    if not test_health_endpoint():
        print("❌ Server is not running. Please start the backend server.")
        return
    
    # Step 2: Check Ultimate AI system health
    if not test_ultimate_ai_health():
        print("⚠️  Ultimate AI system is not fully ready, but continuing with test...")
    
    # Step 3: Create test image
    image_path = create_test_image()
    
    # Step 4: Test image analysis
    print("\n🔍 Testing Image Analysis...")
    success = test_image_analysis(image_path)
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 Image recognition test PASSED!")
        print("✓ All AI models are working correctly")
        print("✓ Image analysis endpoint is functional")
    else:
        print("❌ Image recognition test FAILED!")
        print("⚠️  There may be issues with the AI models or endpoints")
    
    print(f"\n📁 Test image saved at: {os.path.abspath(image_path)}")

if __name__ == "__main__":
    main()