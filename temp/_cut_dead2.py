"""Second cutter pass: library dead functions (encoding-safe UTF-8 IO)."""

import re

TARGETS = {
    "woffl/flow/jetflow.py": ["throat_discharge_old"],
    "woffl/flow/outflow.py": ["homo_diff_press", "production_bottom_up_press"],
    "woffl/assembly/well_sort_client.py": ["_latest"],
}
TOP_LEVEL = re.compile(r"^(def |class |@)")

for path, names in TARGETS.items():
    with open(path, encoding="utf-8", newline="") as fh:
        lines = fh.readlines()
    for name in names:
        start = next(
            (i for i, ln in enumerate(lines) if ln.startswith(f"def {name}(")), None
        )
        assert start is not None, f"{name} not found in {path}"
        end = next(
            (j for j in range(start + 1, len(lines)) if TOP_LEVEL.match(lines[j])),
            len(lines),
        )
        cut_start = start
        while cut_start > 0 and lines[cut_start - 1].strip() == "":
            cut_start -= 1
        removed = end - cut_start
        del lines[cut_start:end]
        if cut_start != len(lines) and cut_start > 0:
            lines[cut_start:cut_start] = ["\n", "\n"]
        print(f"{path}: cut {name} ({removed} lines)")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(lines)
print("done")
