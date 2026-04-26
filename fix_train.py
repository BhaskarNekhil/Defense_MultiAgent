lines = open('train.py', encoding='utf-8').readlines()
for i, line in enumerate(lines):
    if 'header' in line and 'True' in line and 'Pred' in line and 'f"' in line:
        lines[i] = (
            '    _col_label = "True / Pred"\n'
            '    header = f"  {_col_label:<26}" + "".join(f" {abbrev.get(c,c[:4]):>5}" for c in col_order)\n'
        )
        print(f"Fixed line {i+1}")
        break
open('train.py', 'w', encoding='utf-8').writelines(lines)
print("Done!")
