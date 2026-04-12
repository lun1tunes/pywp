import os
import re

FILES = [
    "pywp/actual_fund_analysis.py",
    "pywp/anticollision.py",
    "pywp/ptc_core.py",
    "pywp/well_pad.py",
    "pywp/reference_trajectories.py",
    "pywp/planner.py"
]

def add_import(content: str) -> str:
    if "from pywp.constants import SMALL" in content or "from pywp.constants import" in content and "SMALL" in content:
        return content
    # Find last from __future__ or standard import and add
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("from pywp.models"):
            lines.insert(i, "from pywp.constants import SMALL")
            return "\n".join(lines)
    return content

for file in FILES:
    with open(file, "r") as f:
        content = f.read()
    
    if "1e-9" in content:
        content = add_import(content)
        content = content.replace("1e-9", "SMALL")
        
        with open(file, "w") as f:
            f.write(content)
