"""统计源码 / 测试规模，用于对照"2~3 天能读透"的目标。

同时区分**代码行**与**注释行**：本项目刻意保留了大量"为什么这么做"的注释
（记录踩过的坑），所以总行数会明显大于纯逻辑行数。两个数字都给出，
避免"看起来 3700 行很重"这种误判。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def measure(paths):
    files = [p for p in paths if p.is_file()]
    lines = sum(len(p.read_text(encoding="utf-8", errors="ignore").splitlines()) for p in files)
    return len(files), lines


def logic_lines(path: Path) -> int:
    """统计"有效代码行"：排除空行、整行注释、以及文档字符串。"""
    import ast

    source = path.read_text(encoding="utf-8", errors="ignore")
    doc_lines: set[int] = set()
    try:
        tree = ast.parse(source)
        # 命名约定 `_ignored_docstring = '''...'''`：用来留住一段暂时不想删、
        # 又不希望被计入代码行的说明文字。这里按约定跳过。
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.targets:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("_ignored_docstring"):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        for lineno in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                            doc_lines.add(lineno)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    if isinstance(body[0].value.value, str):
                        for lineno in range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1):
                            doc_lines.add(lineno)
    except SyntaxError:
        pass

    count = 0
    for index, raw in enumerate(source.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or index in doc_lines:
            continue
        count += 1
    return count


def main() -> int:
    groups = [
        ("源码 docpilot/", list((ROOT / "docpilot").rglob("*.py"))),
        ("测试 tests/", list((ROOT / "tests").rglob("*.py"))),
        ("脚本 scripts/", list((ROOT / "scripts").rglob("*.py"))),
    ]
    print(f"{'目录':<18}{'文件':>5}{'总行数':>8}{'代码行':>8}")
    print("-" * 41)
    for name, paths in groups:
        count, lines = measure(paths)
        logic = sum(logic_lines(p) for p in paths if p.is_file())
        print(f"{name:<18}{count:>5}{lines:>8}{logic:>8}")

    source_paths = groups[0][1]
    _, source_lines = measure(source_paths)
    source_logic = sum(logic_lines(p) for p in source_paths if p.is_file())
    print()
    print(f"源码合计：{source_lines} 行（其中**有效代码 {source_logic} 行**，其余为解释性注释与文档）")
    print("目标：1500~2000 行可读透")
    verdict = "达标" if source_logic <= 2200 else "偏多，需要精简"
    print(f"判定：按有效代码行看 → {verdict}")

    print("\n各模块规模：")
    print(f"  {'模块':<24}{'总行':>6}{'代码行':>8}")
    for path in sorted((ROOT / "docpilot").rglob("*.py")):
        lines = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        print(f"  {path.relative_to(ROOT / 'docpilot').as_posix():<24}{lines:>6}{logic_lines(path):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
