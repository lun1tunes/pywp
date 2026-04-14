with open("pywp/three_viewer_assets/templates/viewer_template.html") as f:
    for i, line in enumerate(f):
        if "function renderCollisions" in line:
            for j in range(30):
                print(next(f).rstrip())
            break
