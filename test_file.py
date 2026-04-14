import re
with open("pywp/three_viewer_assets/templates/viewer_template.html") as f:
    t = f.read()

print("syncLegendVisibility();" in t)
print("renderCollisions()" in t)

