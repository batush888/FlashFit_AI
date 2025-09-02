#!/usr/bin/env python3
"""
Test script to debug the suggestions endpoint issue
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8080"
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "password123"

def get_auth_token():
    """Get authentication token"""
    login_data = {
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"Login failed: {response.status_code} - {response.text}")
        return None

def get_wardrobe_items(token):
    """Get user's wardrobe items"""
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/api/wardrobe", headers=headers)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Failed to get wardrobe: {response.status_code} - {response.text}")
        return []

def test_match_endpoint(token, item_id):
    """Test the match endpoint with a specific item ID"""
    headers = {"Authorization": f"Bearer {token}"}
    
    match_data = {
        "item_id": item_id,
        "occasion": "casual",
        "target_count": 3
    }
    
    print(f"\nTesting /api/match with item_id: {item_id}")
    print(f"Request data: {json.dumps(match_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/match", json=match_data, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response.status_code == 200

def test_with_invalid_item_id(token):
    """Test with invalid item ID (like 'default')"""
    headers = {"Authorization": f"Bearer {token}"}
    
    match_data = {
        "item_id": "default",
        "occasion": "casual",
        "target_count": 3
    }
    
    print(f"\nTesting /api/match with invalid item_id: 'default'")
    print(f"Request data: {json.dumps(match_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/match", json=match_data, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response.status_code

def test_with_real_item_id(token):
    """Test with a known real item ID"""
    headers = {"Authorization": f"Bearer {token}"}
    
    # Use a known item ID from the users.json file
    match_data = {
        "item_id": "item_964537a9a9a44e89aa2d5bfd3bd5d9b2",
        "occasion": "casual",
        "target_count": 3
    }
    
    print(f"\nTesting /api/match with real item_id: 'item_964537a9a9a44e89aa2d5bfd3bd5d9b2'")
    print(f"Request data: {json.dumps(match_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/match", json=match_data, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response.status_code == 200

def main():
    print("=== Testing Suggestions Endpoint ===")
    
    # Get authentication token
    print("1. Getting authentication token...")
    token = get_auth_token()
    if not token:
        print("Failed to get authentication token. Exiting.")
        return
    
    print(f"✓ Got token: {token[:20]}...")
    
    # Get wardrobe items
    print("\n2. Getting wardrobe items...")
    wardrobe_items = get_wardrobe_items(token)
    print(f"Found {len(wardrobe_items)} wardrobe items")
    
    if wardrobe_items:
        for i, item in enumerate(wardrobe_items[:3]):  # Show first 3 items
            print(f"  Item {i+1}: {item.get('item_id', 'N/A')} - {item.get('garment_type', 'Unknown')}")
    
    # Test with invalid item ID (reproducing frontend issue)
    print("\n3. Testing with invalid item_id 'default'...")
    status_code = test_with_invalid_item_id(token)
    
    if status_code == 400:
        print("✓ Confirmed: 'default' item_id causes 400 Bad Request")
    
    # Test with known real item ID
    print("\n4. Testing with known real item ID...")
    success = test_with_real_item_id(token)
    if success:
        print("✓ Success with real item_id")
    else:
        print("✗ Failed with real item_id")
    
    # Test with valid item IDs if available
    if wardrobe_items:
        print("\n5. Testing with wardrobe item IDs...")
        for item in wardrobe_items[:2]:  # Test first 2 items
            item_id = item.get('item_id')
            if item_id:
                success = test_match_endpoint(token, item_id)
                if success:
                    print(f"✓ Success with item_id: {item_id}")
                else:
                    print(f"✗ Failed with item_id: {item_id}")
    else:
        print("\n5. No wardrobe items found to test with")
        print("   Upload some items first to test suggestions")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()