"""Run every example in order and regenerate all figures in figures/."""
import runpy
import sys
from pathlib import Path

examples = sorted((Path(__file__).parent / "examples").glob("0*.py"))
sys.path.insert(0, str(examples[0].parent))
for script in examples:
    print(f"\n=== {script.name} ===")
    runpy.run_path(str(script), run_name="__main__")
