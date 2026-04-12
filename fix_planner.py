import re

with open("pywp/planner.py", "r") as f:
    content = f.read()

# 1. Remove _validate_config function definition
content = re.sub(r'def _validate_config\(config: TrajectoryConfig\) -> None:.*?for segment, limit in config\.dls_limits_deg_per_30m\.items\(\):\n        if limit < 0\.0:\n            raise PlanningError\(f"DLS limit for segment \{segment\} cannot be negative\."\)\n\n', '', content, flags=re.DOTALL)

# 2. Replace the call
content = content.replace('_validate_config(config)', 'config.validate_for_planning()')

# 3. Replace the geometry method
content = content.replace(
    '        zero_azimuth_turn = _is_zero_azimuth_turn_geometry(\n            geometry=geometry,\n            target_direction=target_direction,\n            tolerance_m=float(config.lateral_tolerance_m),\n        )',
    '        zero_azimuth_turn = geometry.is_zero_azimuth_turn(\n            target_direction=target_direction,\n            tolerance_m=float(config.lateral_tolerance_m),\n        )'
)

# 4. Remove unused imports in planner.py
imports_to_remove = [
    '    _azimuth_deg_from_pair,\n',
    '    _azimuth_deg_from_points,\n',
    '    _distance_3d,\n',
    '    _inclination_from_displacement,\n',
    '    _is_geometry_coplanar,\n',
    '    _is_zero_azimuth_turn_geometry,\n',
    '    _project_to_section_axis,\n',
    '    ProfileEndpointEvaluation,\n'
]
for imp in imports_to_remove:
    content = content.replace(imp, '')

# Write back
with open("pywp/planner.py", "w") as f:
    f.write(content)
