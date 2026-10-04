import ast

from clinical_deid.paths import PACKAGE_ROOT

RESEARCH = ("clinical_deid.data", "clinical_deid.training", "clinical_deid.evaluation")
RESEARCH_DIRS = {"data", "training", "evaluation"}


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_runtime_modules_do_not_import_research_code():
    violations = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.relative_to(PACKAGE_ROOT).parts[0] in RESEARCH_DIRS:
            continue
        for module in imported_modules(path):
            if module.startswith(RESEARCH):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)} imports {module}")
    assert violations == []


def test_package_uses_absolute_imports():
    relative = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in PACKAGE_ROOT.rglob("*.py")
        if any(
            isinstance(node, ast.ImportFrom) and node.level
            for node in ast.walk(ast.parse(path.read_text()))
        )
    ]
    assert relative == []
