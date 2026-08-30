"""Put the repo root on sys.path so `src.telco_churn...` imports resolve.

Lets pytest run from anywhere without requiring PYTHONPATH to be set by hand.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
