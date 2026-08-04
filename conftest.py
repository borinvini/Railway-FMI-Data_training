# conftest.py
"""Put the repository root on sys.path so `import src...` and `import config...`
work regardless of how pytest is invoked."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
