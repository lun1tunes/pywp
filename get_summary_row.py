with open("pywp/ptc_core.py") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "summary_rows.append" in line:
        for j in range(i-15, i+5):
            print(f"{j+1}: {lines[j].rstrip()}")
        break
