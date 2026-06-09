"""Find function-local imports that shadow a name used EARLIER in the same
function (the UnboundLocalError class of bug)."""

import ast
import sys

FILES = [
    "woffl/gui/app.py",
    "woffl/gui/sidebar.py",
    "woffl/gui/utils.py",
    "woffl/gui/pdf_export.py",
    "woffl/gui/optimization_viz.py",
    "woffl/gui/workflow_page.py",
    "woffl/gui/workflow_steps/step1_select_wells.py",
    "woffl/gui/workflow_steps/step2_review_ipr.py",
    "woffl/gui/workflow_steps/step2_5_precalibrate.py",
    "woffl/gui/workflow_steps/step3_configure_optimize.py",
    "woffl/gui/workflow_steps/step4_results.py",
    "woffl/gui/tabs/batch_run.py",
    "woffl/gui/tabs/jetpump_solver.py",
    "woffl/gui/tabs/pressure_profile.py",
    "woffl/gui/scotts_tools/_common.py",
    "woffl/gui/scotts_tools/header_impact.py",
    "woffl/gui/scotts_tools/header_engine.py",
    "woffl/gui/scotts_tools/pf_scenario.py",
    "woffl/gui/scotts_tools/well_sort.py",
    "woffl/gui/scotts_tools/jp_calibration.py",
    "woffl/gui/scotts_tools/jp_fric_trend.py",
    "woffl/gui/scotts_tools/jp_washout.py",
]

problems = []
for path in FILES:
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # names bound by imports inside this function (excluding nested defs)
        local_imports = {}  # name -> lineno of import
        loads = []  # (name, lineno)
        for node in ast.walk(func):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func:
                continue  # shallow enough for this check
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    if name not in local_imports or node.lineno < local_imports[name]:
                        local_imports[name] = node.lineno
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loads.append((node.id, node.lineno))
        for name, line in loads:
            if name in local_imports and line < local_imports[name]:
                problems.append(
                    f"{path}:{line} '{name}' used before its function-local "
                    f"import at line {local_imports[name]} (in {func.name})"
                )

if problems:
    print("\n".join(problems))
    sys.exit(1)
print(f"checked {len(FILES)} files: no use-before-local-import shadowing found")
