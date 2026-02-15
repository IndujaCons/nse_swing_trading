#!/usr/bin/env python3
"""
Application launcher for RS Dashboard
"""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(__file__))

from ui.app import run_server

if __name__ == "__main__":
    run_server()
