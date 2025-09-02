#!/usr/bin/env python3
import requests
import json

# Test the profile endpoint with the token
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyXzFfMTc1NjgxOTAwNyIsImV4cCI6MTc1NzQyNDE3MX0.Hk0r_ttGTSGe-Cjch9t_vDKrK9G8mvRqXlj6ZnINCE8"

headers = {
    "Authorization": f"Bearer {token}"
}

print("Testing /api/user/profile endpoint...")
response = requests.get('http://localhost:8080/api/user/profile', headers=headers)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code != 200:
    print("\nTesting without token...")
    response_no_auth = requests.get('http://localhost:8080/api/user/profile')
    print(f"Status Code (no auth): {response_no_auth.status_code}")
    print(f"Response (no auth): {response_no_auth.text}")