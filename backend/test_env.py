#!/usr/bin/env python3

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("Testing environment variables:")
print(f"OPENAI_API_KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME')}")
print(f"NEO4J_PASSWORD exists: {bool(os.getenv('NEO4J_PASSWORD'))}")

# Test config module
from config import Config

try:
    Config.validate()
    print("\nConfig validation: SUCCESS")
except ValueError as e:
    print(f"\nConfig validation failed: {e}")