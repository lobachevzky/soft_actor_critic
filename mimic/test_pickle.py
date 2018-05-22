#! /usr/bin/env python
import pickle
import sys
from pathlib import Path

with Path(sys.argv[1]).open('rb') as f:
    print(pickle.load(f))
