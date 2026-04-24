import subprocess
import sys

result = subprocess.run(
    [sys.executable, "tools/optuna_search_classifier.py", 
     "--n-trials", "50", 
     "--epochs", "2", 
     "--sample-size", "2000", 
     "--study-name", "unit_classifier_search_v2"],
    capture_output=True,
    text=True,
    cwd="."
)

print("=== STDOUT ===")
print(result.stdout)
print("\n=== STDERR ===")
print(result.stderr)
print("\n=== Return Code ===")
print(result.returncode)
