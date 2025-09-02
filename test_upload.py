#!/usr/bin/env python3
import requests
import os
from io import BytesIO
from PIL import Image

# Test the upload endpoint with authentication
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyXzFfMTc1NjgxOTAwNyIsImV4cCI6MTc1NzQyNDE3MX0.Hk0r_ttGTSGe-Cjch9t_vDKrK9G8mvRqXlj6ZnINCE8"

headers = {
    "Authorization": f"Bearer {token}"
}

print("Testing /api/upload endpoint...")

# Create a simple test image using PIL
test_image = Image.new('RGB', (100, 100), color='red')
img_buffer = BytesIO()
test_image.save(img_buffer, format='JPEG')
img_buffer.seek(0)

# Test upload with authentication
files = {'file': ('test_image.jpg', img_buffer, 'image/jpeg')}

response = requests.post('http://localhost:8080/api/upload', 
                        headers=headers, 
                        files=files)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    print("✓ Upload successful with authentication!")
else:
    print("✗ Upload failed")
    
    # Test without authentication to compare
    print("\nTesting without authentication...")
    img_buffer.seek(0)
    files = {'file': ('test_image.jpg', img_buffer, 'image/jpeg')}
    response_no_auth = requests.post('http://localhost:8080/api/upload', files=files)
    print(f"Status Code (no auth): {response_no_auth.status_code}")
    print(f"Response (no auth): {response_no_auth.text}")