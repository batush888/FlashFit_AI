#!/usr/bin/env python3
import requests
import json

# Login to get token
login_data = {
    "email": "test@example.com",
    "password": "password123"
}

response = requests.post('http://localhost:8080/api/auth/login', json=login_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    data = response.json()
    token = data.get('data', {}).get('token')
    print(f"\nToken: {token}")
    print(f"\nTo use in browser console:")
    print(f"localStorage.setItem('auth_token', '{token}');")
else:
    print("Login failed")