"""Run the current hybrid classifier training entrypoint."""
import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, 'training/train_unit_classifier_hybrid.py']))
