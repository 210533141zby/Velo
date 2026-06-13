"""把实验脚本同步生成为可直接查阅的 notebook。

生成目标：
1. 每个实验脚本都有一个同名 .ipynb；
2. notebook 中保留完整源码；
3. notebook 中直接展示对应结果文件的预览，不必重新运行也能查看。
"""

from __future__ import annotations

import csv
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PYTHON_BIN = "/root/Velo/.venv/bin/python"


@dataclass(frozen=True)
class ResultEntry:
    title: str
    path: Path
    note: str = ""


@dataclass(frozen=True)
class NotebookSpec:
    title: str
    thesis_section: str
    script_path: Path
    intro: str
    run_example: str
    results: tuple[ResultEntry, ...]


def relpath(from_dir: Path, target: Path) -> str:
    """计算从 notebook 所在目录指向结果文件的相对路径。"""
    return os.path.relpath(target, start=from_dir)


def to_posix_rel(from_dir: Path, target: Path) -> str:
    """把相对路径统一转换成 POSIX 形式，便于 notebook 稳定引用。"""
    return Path(relpath(from_dir, target)).as_posix()


def preview_json_data(data: Any) -> Any:
    """把完整 JSON 结果压缩成适合 notebook 展示的预览结构。

    如果结果列表过长，这里只保留总数和前若干项，
    让 notebook 在不丢失整体轮廓的前提下保持可读。
    """
    if isinstance(data, list):
        if len(data) <= 6:
            return data
        return {"total_items": len(data), "preview": data[:6]}
    if isinstance(data, dict):
        preview = dict(data)
        for key in ("summaries", "preview", "rows"):
            value = preview.get(key)
            if isinstance(value, list) and len(value) > 6:
                preview[key] = {
                    "total_items": len(value),
                    "preview": value[:6],
                }
        return preview
    return data


def json_preview_text(path: Path) -> str:
    """读取 JSON 结果文件，并生成一份适合 notebook 展示的文本预览。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    preview = preview_json_data(data)
    return json.dumps(preview, ensure_ascii=False, indent=2)


def csv_preview(path: Path, *, limit: int = 8) -> tuple[str, str]:
    """把 CSV 结果文件压成纯文本与 HTML 双格式预览。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    preview_rows = rows[:limit]
    plain_lines = [f"rows={len(rows)} preview={len(preview_rows)}"]
    if preview_rows:
        headers = list(preview_rows[0].keys())
        header_line = " | ".join(headers)
        plain_lines.append(header_line)
        plain_lines.append("-" * len(header_line))
        for row in preview_rows:
            plain_lines.append(" | ".join(str(row.get(header, "")) for header in headers))
    else:
        headers = []
        plain_lines.append("(empty)")

    html_parts = [f"<p><strong>rows={len(rows)} preview={len(preview_rows)}</strong></p>"]
    if preview_rows:
        html_parts.append("<table>")
        html_parts.append("<thead><tr>")
        for header in headers:
            html_parts.append(f"<th>{html.escape(header)}</th>")
        html_parts.append("</tr></thead><tbody>")
        for row in preview_rows:
            html_parts.append("<tr>")
            for header in headers:
                html_parts.append(f"<td>{html.escape(str(row.get(header, '')))}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")
    return "\n".join(plain_lines), "".join(html_parts)


def build_json_preview_cell(script_dir: Path, entry: ResultEntry, execution_count: int):
    """为 JSON 结果文件生成一格带静态输出的 notebook 代码单元。"""
    rel_file = to_posix_rel(script_dir, entry.path)
    source = (
        "from pathlib import Path\n"
        "import json\n\n"
        f"path = Path({rel_file!r})\n"
        "data = json.loads(path.read_text(encoding='utf-8'))\n"
        "if isinstance(data, list) and len(data) > 6:\n"
        "    data = {'total_items': len(data), 'preview': data[:6]}\n"
        "elif isinstance(data, dict):\n"
        "    data = dict(data)\n"
        "    for key in ('summaries', 'preview', 'rows'):\n"
        "        value = data.get(key)\n"
        "        if isinstance(value, list) and len(value) > 6:\n"
        "            data[key] = {'total_items': len(value), 'preview': value[:6]}\n"
        "print(json.dumps(data, ensure_ascii=False, indent=2))\n"
    )
    output_text = json_preview_text(entry.path)
    return nbf.v4.new_code_cell(
        source=source,
        execution_count=execution_count,
        outputs=[nbf.v4.new_output("stream", name="stdout", text=output_text + "\n")],
    )


def build_csv_preview_cell(script_dir: Path, entry: ResultEntry, execution_count: int):
    """为 CSV 结果文件生成一格带表格预览的 notebook 代码单元。"""
    rel_file = to_posix_rel(script_dir, entry.path)
    source = (
        "from pathlib import Path\n"
        "import csv\n\n"
        f"path = Path({rel_file!r})\n"
        "with path.open('r', encoding='utf-8', newline='') as handle:\n"
        "    rows = list(csv.DictReader(handle))\n"
        "print(f'rows={len(rows)} preview={min(len(rows), 8)}')\n"
        "for row in rows[:8]:\n"
        "    print(row)\n"
    )
    plain_text, html_table = csv_preview(entry.path)
    return nbf.v4.new_code_cell(
        source=source,
        execution_count=execution_count,
        outputs=[
            nbf.v4.new_output(
                "display_data",
                data={"text/plain": plain_text, "text/html": html_table},
            )
        ],
    )


def build_svg_preview_cell(script_dir: Path, entry: ResultEntry):
    """为 SVG 图像结果生成 markdown 预览单元，便于直接内嵌查看。"""
    rel_file = to_posix_rel(script_dir, entry.path)
    note = f"\n\n{entry.note}" if entry.note else ""
    return nbf.v4.new_markdown_cell(
        f"**图像文件**：`{rel_file}`{note}\n\n![{entry.title}]({rel_file})"
    )


def result_entry_cells(script_dir: Path, entry: ResultEntry, execution_count: int) -> tuple[list[Any], int]:
    """根据结果文件类型生成对应的 notebook 单元列表。"""
    cells: list[Any] = [
        nbf.v4.new_markdown_cell(
            f"### {entry.title}\n\n- 文件：`{to_posix_rel(script_dir, entry.path)}`"
            + (f"\n- 说明：{entry.note}" if entry.note else "")
        )
    ]
    suffix = entry.path.suffix.lower()
    if suffix == ".json":
        cells.append(build_json_preview_cell(script_dir, entry, execution_count))
        execution_count += 1
    elif suffix == ".csv":
        cells.append(build_csv_preview_cell(script_dir, entry, execution_count))
        execution_count += 1
    elif suffix == ".svg":
        cells.append(build_svg_preview_cell(script_dir, entry))
    else:
        cells.append(nbf.v4.new_markdown_cell("当前 notebook 只内嵌预览常见结果文件，请直接打开原文件查看。"))
    return cells, execution_count


def build_notebook(spec: NotebookSpec) -> None:
    """根据规格生成一份可直接查阅的 notebook。

    生成结果会同时保留源码镜像、命令行复现方式和已保存结果预览，
    方便答辩或代码走查时直接打开查看，不必现场重新跑实验。
    """
    script_dir = spec.script_path.parent
    notebook_path = spec.script_path.with_suffix(".ipynb")
    script_text = spec.script_path.read_text(encoding="utf-8")

    cells: list[Any] = [
        nbf.v4.new_markdown_cell(
            f"# {spec.title}\n\n"
            f"- 对应论文章节：{spec.thesis_section}\n"
            f"- 源脚本：`{spec.script_path.relative_to(REPO_ROOT).as_posix()}`\n"
            f"- notebook 作用：直接查看代码与已保存结果，命令行运行仍以 `.py` 为准\n\n"
            f"{spec.intro}"
        ),
        nbf.v4.new_markdown_cell(
            "## 命令行复现\n\n"
            "```bash\n"
            f"cd {REPO_ROOT}\n"
            f"{PYTHON_BIN} {spec.run_example}\n"
            "```"
        ),
        nbf.v4.new_markdown_cell("## 源码镜像\n\n下面这一格保留 `.py` 的完整源码，主要用于现场查阅。"),
        nbf.v4.new_code_cell(source=script_text, execution_count=None, outputs=[]),
        nbf.v4.new_markdown_cell("## 结果预览\n\n下面直接内嵌当前已保存结果的关键文件预览。"),
    ]

    execution_count = 1
    for entry in spec.results:
        entry_cells, execution_count = result_entry_cells(script_dir, entry, execution_count)
        cells.extend(entry_cells)

    notebook = nbf.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook_path.write_text(nbf.writes(notebook), encoding="utf-8")


def build_specs() -> list[NotebookSpec]:
    """列出需要同步转换成 notebook 的实验脚本与结果文件映射。"""
    return [
        NotebookSpec(
            title="主线四档消融",
            thesis_section="第3章 RAG算法实验分析",
            script_path=ROOT / "02_消融实验/scripts/运行_主线四档消融.py",
            intro="这本 notebook 对应论文主线从基线到“分项重排 + 覆盖取证 + 按题作答”的四档提升过程。",
            run_example="experiments/02_消融实验/scripts/运行_主线四档消融.py",
            results=(
                ResultEntry("评测批次说明", ROOT / "02_消融实验/results/01_主线四档消融_20260517/评测批次说明.json"),
                ResultEntry("消融实验汇总", ROOT / "02_消融实验/results/01_主线四档消融_20260517/消融实验_汇总.json"),
                ResultEntry("消融实验指标表（JSON）", ROOT / "02_消融实验/results/01_主线四档消融_20260517/消融实验_指标表.json"),
                ResultEntry("消融实验指标表（CSV）", ROOT / "02_消融实验/results/01_主线四档消融_20260517/消融实验_指标表.csv"),
            ),
        ),
        NotebookSpec(
            title="最终参数搜索",
            thesis_section="第3章 RAG算法实验分析",
            script_path=ROOT / "02_消融实验/scripts/运行_最终参数搜索.py",
            intro="这本 notebook 对应 4/6、5/7 等参数组合比较，用于说明最终参数配置的选择过程。",
            run_example="experiments/02_消融实验/scripts/运行_最终参数搜索.py",
            results=(
                ResultEntry("评测批次说明", ROOT / "02_消融实验/results/02_最终参数搜索_20260517/评测批次说明.json"),
                ResultEntry("参数搜索路线总表", ROOT / "02_消融实验/results/02_最终参数搜索_20260517/参数搜索_路线总表.json"),
                ResultEntry("参数搜索全部汇总", ROOT / "02_消融实验/results/02_最终参数搜索_20260517/参数搜索_全部汇总.json"),
            ),
        ),
        NotebookSpec(
            title="RGB300支撑评测",
            thesis_section="第3章 RAG算法实验分析",
            script_path=ROOT / "02_消融实验/scripts/运行_RGB300支撑评测.py",
            intro="这本 notebook 对应 RGB300 稳定性支撑实验，以及和 CRUD 对照的图表输出。",
            run_example="experiments/02_消融实验/scripts/运行_RGB300支撑评测.py",
            results=(
                ResultEntry("实验配置", ROOT / "02_消融实验/results/03_RGB300支撑结果/支撑实验_实验配置.json"),
                ResultEntry("实验汇总", ROOT / "02_消融实验/results/03_RGB300支撑结果/支撑实验_汇总.json"),
                ResultEntry("RGB300 指标表", ROOT / "02_消融实验/results/03_RGB300支撑结果/RGB300_指标表.csv"),
                ResultEntry("CRUD 对照指标表", ROOT / "02_消融实验/results/03_RGB300支撑结果/CRUD对照_指标表.csv"),
                ResultEntry("RGB300 柱状图", ROOT / "02_消融实验/results/03_RGB300支撑结果/RGB300_柱状图.svg"),
                ResultEntry("CRUD 对照柱状图", ROOT / "02_消融实验/results/03_RGB300支撑结果/CRUD对照_柱状图.svg"),
            ),
        ),
        NotebookSpec(
            title="主线替代路线对比",
            thesis_section="第3.3.1节 主线替代路线",
            script_path=ROOT / "03_证伪实验/scripts/运行_主线替代路线对比.py",
            intro="这本 notebook 统一展示双片段证据保留、证据标注、局部窗口改写、模板引导、支持表综合等主线附近替代路线。",
            run_example="experiments/03_证伪实验/scripts/运行_主线替代路线对比.py --suite all",
            results=(
                ResultEntry("双片段路线：评测批次说明", ROOT / "03_证伪实验/results/01_主线替代路线/双片段路线_20260517/评测批次说明.json"),
                ResultEntry("双片段路线：汇总", ROOT / "03_证伪实验/results/01_主线替代路线/双片段路线_20260517/主线替代路线_汇总.json"),
                ResultEntry("双片段路线：指标表", ROOT / "03_证伪实验/results/01_主线替代路线/双片段路线_20260517/主线替代路线_指标表.csv"),
                ResultEntry("候选证据路线：评测批次说明", ROOT / "03_证伪实验/results/01_主线替代路线/候选证据路线_20260517/评测批次说明.json"),
                ResultEntry("候选证据路线：对比总表", ROOT / "03_证伪实验/results/01_主线替代路线/候选证据路线_20260517/对比总表.json"),
                ResultEntry("结构化生成路线：评测批次说明", ROOT / "03_证伪实验/results/01_主线替代路线/结构化生成路线_20260517/评测批次说明.json"),
                ResultEntry("结构化生成路线：对比总表", ROOT / "03_证伪实验/results/01_主线替代路线/结构化生成路线_20260517/对比总表.json"),
            ),
        ),
        NotebookSpec(
            title="生成侧提示路线对比",
            thesis_section="第3.3.2节 生成侧补充对比实验",
            script_path=ROOT / "03_证伪实验/scripts/运行_生成侧提示路线对比.py",
            intro="这本 notebook 对应少样例风格锚定、自由式思维展开，以及基线直接作答三条生成侧提示路线。",
            run_example="experiments/03_证伪实验/scripts/运行_生成侧提示路线对比.py --variant baseline_prompt4",
            results=(
                ResultEntry("生成侧：基线直接作答", ROOT / "03_证伪实验/results/02_生成侧补充对比/生成侧_基线直接作答_结果汇总.json"),
                ResultEntry("生成侧：少样例风格锚定", ROOT / "03_证伪实验/results/02_生成侧补充对比/生成侧_少样例风格锚定_结果汇总.json"),
                ResultEntry("生成侧：自由式思维展开", ROOT / "03_证伪实验/results/02_生成侧补充对比/生成侧_自由式思维展开_结果汇总.json"),
            ),
        ),
        NotebookSpec(
            title="支持修补对比",
            thesis_section="第3.3.2节 生成侧补充对比实验",
            script_path=ROOT / "03_证伪实验/scripts/运行_支持修补对比.py",
            intro="这本 notebook 单独展示生成后支持性修补和基线、对齐作答之间的差异。",
            run_example="experiments/03_证伪实验/scripts/运行_支持修补对比.py",
            results=(
                ResultEntry("支持性修补对比结果", ROOT / "03_证伪实验/results/02_生成侧补充对比/生成侧_支持性修补对比_结果汇总.json"),
            ),
        ),
        NotebookSpec(
            title="检索侧补充对比",
            thesis_section="第3.3.3节 检索侧补充对比实验",
            script_path=ROOT / "03_证伪实验/scripts/运行_检索侧补充对比.py",
            intro="这本 notebook 对应上下文化检索、查询改写、父子块检索与基线之间的比较。",
            run_example="experiments/03_证伪实验/scripts/运行_检索侧补充对比.py --variant baseline",
            results=(
                ResultEntry("检索侧：基线", ROOT / "03_证伪实验/results/03_检索侧补充对比/检索侧_基线_结果汇总.json"),
                ResultEntry("检索侧：上下文化检索", ROOT / "03_证伪实验/results/03_检索侧补充对比/检索侧_上下文化检索_结果汇总.json"),
                ResultEntry("检索侧：查询改写", ROOT / "03_证伪实验/results/03_检索侧补充对比/检索侧_查询改写_结果汇总.json"),
                ResultEntry("检索侧：父子块检索", ROOT / "03_证伪实验/results/03_检索侧补充对比/检索侧_父子块检索_结果汇总.json"),
            ),
        ),
        NotebookSpec(
            title="回答模型对比",
            thesis_section="补充答辩材料：回答模型比较",
            script_path=ROOT / "03_证伪实验/scripts/运行_回答模型对比.py",
            intro="这本 notebook 固定检索链路，只比较不同回答模型和提示口径。",
            run_example="experiments/03_证伪实验/scripts/运行_回答模型对比.py --models qwen2.5:7b-instruct qwen3:8b llama3.1:8b",
            results=(
                ResultEntry("回答模型对比结果", ROOT / "03_证伪实验/results/04_回答模型对比/回答模型对比_结果汇总_qwen25_qwen3_llama31_neutral_native.json"),
            ),
        ),
        NotebookSpec(
            title="TRACE方法对比",
            thesis_section="补充答辩材料：TRACE 方法对比",
            script_path=ROOT / "03_证伪实验/scripts/运行_TRACE方法对比.py",
            intro="这本 notebook 对应 TRACE 方法和论文主线方案之间的直接对比。",
            run_example="experiments/03_证伪实验/scripts/运行_TRACE方法对比.py",
            results=(
                ResultEntry("评测批次说明", ROOT / "03_证伪实验/results/05_TRACE方法对比/TRACE方法对比_20260520/评测批次说明.json"),
                ResultEntry("TRACE 对比汇总", ROOT / "03_证伪实验/results/05_TRACE方法对比/TRACE方法对比_20260520/TRACE对比_汇总.json"),
                ResultEntry("TRACE 对比指标表（JSON）", ROOT / "03_证伪实验/results/05_TRACE方法对比/TRACE方法对比_20260520/TRACE对比_指标表.json"),
                ResultEntry("TRACE 对比指标表（CSV）", ROOT / "03_证伪实验/results/05_TRACE方法对比/TRACE方法对比_20260520/TRACE对比_指标表.csv"),
            ),
        ),
    ]


def main() -> None:
    """组织当前脚本的主执行流程。"""
    specs = build_specs()
    for spec in specs:
        build_notebook(spec)
        print(f"generated: {spec.script_path.with_suffix('.ipynb').relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
