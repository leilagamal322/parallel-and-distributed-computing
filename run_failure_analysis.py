#!/usr/bin/env python3
"""
Runner script for failure performance analysis
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from failure_performance_analysis import main

if __name__ == "__main__":
    main()

