with open("pywp/welltrack_batch.py") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "def _execute_single" in line:
        for j in range(i, i+60):
            print(lines[j].rstrip())
        break
