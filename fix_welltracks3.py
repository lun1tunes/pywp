with open("tests/test_data/WELLTRACKS3.INC", "r") as f:
    content = f.read()

content = content.replace(
"""WELLTRACK 'well_02'
598863.0\t7411139.0\t0.0\t0
597390.0\t7408694.0\t3766.209\t3767
599263.0\t7409850.0\t3759.833\t5970""",
"""WELLTRACK 'well_02'
598863.0\t7411139.0\t0.0\t0
599263.0\t7409850.0\t3759.833\t3767
597390.0\t7408694.0\t3766.209\t5970"""
)

content = content.replace(
"""WELLTRACK 'well_04'
598863.0\t7411139.0\t0.0\t0
601041.0\t7412188.0\t3798.500\t3799
599168.0\t7411032.0\t3799.701\t6000""",
"""WELLTRACK 'well_04'
598863.0\t7411139.0\t0.0\t0
599168.0\t7411032.0\t3799.701\t3799
601041.0\t7412188.0\t3798.500\t6000"""
)

with open("tests/test_data/WELLTRACKS3.INC", "w") as f:
    f.write(content)
