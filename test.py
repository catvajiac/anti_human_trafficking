#!/usr/bin/env python3
# Author: Catalina Vajiac
# Purpose: run all tests for relevant code
# Usage: ./test.py

import os, sys
from tests import test_eigenspokes
import unittest


def usage(code):
    print('Usage: {}'.format(os.path.basename(sys.argv[0])))
    exit(code)


if __name__ == '__main__':
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        unittest.main(test_eigenspokes, buffer=False)
