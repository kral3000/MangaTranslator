import ast
from pathlib import Path
from typing import Optional


SOURCE = Path(__file__).resolve().parents[1] / "core/services/translation.py"


def _load_prompt_builders():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    wanted = {"_build_system_prompt_ocr", "_build_system_prompt_translation"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"Optional": Optional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def main() -> None:
    prompts = _load_prompt_builders()
    ocr = prompts["_build_system_prompt_ocr"]("Japanese", "rtl")
    translate = prompts["_build_system_prompt_translation"]
    two_step = translate("Korean", "two-step", "rtl")
    one_step = translate("Korean", "one-step", "rtl")

    assert "must ignore them" not in ocr
    assert "三嶋{みしま}さん！" in ocr
    assert "author's authoritative intended reading" in two_step
    assert "transliterate RUBY" in two_step
    assert "preserve it in the transcription as `BASE{RUBY}`" in one_step


if __name__ == "__main__":
    main()
