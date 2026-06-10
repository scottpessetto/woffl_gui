"""Remove dead top-level functions, encoding-safe (explicit UTF-8 IO)."""

import re

TARGETS = {
    "woffl/gui/workflow_steps/step4_results.py": [
        "_render_pf_sensitivity",
        "_pressure_colors",
    ],
    "woffl/gui/scotts_tools/header_impact.py": ["_ipr_grid_fig", "_ipr_walk_fig"],
    "woffl/gui/scotts_tools/header_engine.py": ["windowed_ipr_fits"],
    "tests/test_header_impact.py": [
        "test_ipr_grid_fig_builds_curve_and_operating_points",
        "test_ipr_grid_fig_none_when_no_jp_rows",
        "test_ipr_grid_fig_overlays_test_points_with_hover",
    ],
    "tests/test_header_engine.py": [
        "test_windowed_ipr_fits_splits_and_tracks_depletion",
        "test_windowed_ipr_fits_too_few",
    ],
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
            (
                j
                for j in range(start + 1, len(lines))
                if TOP_LEVEL.match(lines[j])
            ),
            len(lines),
        )
        # eat the blank separator lines preceding the def (keep exactly the
        # two-blank-line gap owned by the previous block)
        cut_start = start
        while cut_start > 0 and lines[cut_start - 1].strip() == "":
            cut_start -= 1
        removed = end - cut_start
        del lines[cut_start:end]
        # the next block keeps its own leading position; re-insert the
        # standard two blank lines between the previous block and the next
        if cut_start != len(lines) and cut_start > 0:
            lines[cut_start:cut_start] = ["\n", "\n"]
        print(f"{path}: cut {name} ({removed} lines)")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(lines)
print("done")
