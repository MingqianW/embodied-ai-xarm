from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOTS = (
    "simulation",
    "data",
    "evaluation",
    "training",
    "diagnostics",
    "cluster",
    "tools",
    "policy_runtime",
)


def _python_files(*roots: str) -> list[Path]:
    return sorted(
        path
        for root in roots
        for path in (ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _assert_no_imports(roots: tuple[str, ...], forbidden: tuple[str, ...]) -> None:
    violations: list[str] = []
    for path in _python_files(*roots):
        for name in _imports(path):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden):
                violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "Forbidden dependency direction:\n" + "\n".join(violations)


def test_removed_architectures_are_not_imported() -> None:
    _assert_no_imports(PROJECT_ROOTS, ("sim_mujoco", "fine_tune", "slurm"))


def test_canonical_packages_do_not_import_cluster() -> None:
    _assert_no_imports(
        ("simulation", "data", "evaluation", "training", "diagnostics", "policy_runtime", "tools"),
        ("cluster",),
    )


def test_simulation_does_not_depend_on_higher_layers() -> None:
    _assert_no_imports(
        ("simulation",),
        ("data", "evaluation", "training", "diagnostics", "cluster"),
    )


def test_backend_independent_data_does_not_import_simulation() -> None:
    _assert_no_imports(("data/common", "data/real"), ("simulation",))


def test_training_does_not_import_runtime_or_evaluation_layers() -> None:
    _assert_no_imports(
        ("training",),
        ("simulation", "evaluation", "diagnostics", "cluster"),
    )


def test_evaluation_does_not_import_diagnostics() -> None:
    _assert_no_imports(("evaluation",), ("diagnostics",))


def test_policy_runtime_does_not_own_evaluation_results() -> None:
    _assert_no_imports(("policy_runtime",), ("evaluation",))


def _mentions_project_root(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id.endswith("PROJECT_ROOT")
        for child in ast.walk(node)
    )


def _mutates_sys_path_with_project_root(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            is_sys_path = (
                isinstance(owner, ast.Attribute)
                and owner.attr == "path"
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "sys"
            )
            if is_sys_path and node.func.attr in {"insert", "append", "extend"}:
                if any(_mentions_project_root(argument) for argument in node.args):
                    return True
        if isinstance(node, ast.AugAssign) and _mentions_project_root(node.value):
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "path"
                and isinstance(target.value, ast.Name)
                and target.value.id == "sys"
            ):
                return True
    return False


def test_project_code_and_tests_do_not_bootstrap_repository_on_sys_path() -> None:
    violations = [
        path.relative_to(ROOT).as_posix()
        for path in _python_files(*PROJECT_ROOTS, "tests")
        if _mutates_sys_path_with_project_root(path)
    ]
    assert not violations, "Use `python -m package.module` instead: " + ", ".join(violations)


def test_stable_concepts_have_one_definition() -> None:
    files = _python_files(*PROJECT_ROOTS)
    task_registries = []
    lerobot_writers = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == "TASKS" for target in targets):
                    task_registries.append(path.relative_to(ROOT).as_posix())
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "write_xarm_lerobot_dataset":
                lerobot_writers.append(path.relative_to(ROOT).as_posix())
    assert task_registries == ["data/common/task_identity.py"]
    assert lerobot_writers == ["data/common/lerobot_writer.py"]
