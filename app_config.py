# -*- coding: utf-8 -*-

import os
import sys

# Append the local paths to sys.path so we can import them.
PROJECT_ROOT = os.path.abspath(os.path.join('.'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)



