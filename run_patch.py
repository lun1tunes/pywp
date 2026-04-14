import os
path = "pywp/three_viewer_assets/templates/viewer_template.html"
try:
    with open(path, "r") as f:
        html = f.read()
        
    count = html.count("renderCollisions()")
    if count == 1:
        html = html.replace("syncLegendVisibility();", "syncLegendVisibility();\n          renderCollisions();")
        with open(path, "w") as f:
            f.write(html)
        print("Patcher fixed the file.")
    else:
        print(f"Patcher skipped, count is {count}")
except Exception as e:
    print(f"Error: {e}")
