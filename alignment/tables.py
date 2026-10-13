import csv
from pathlib import Path


LATEX_ESCAPES = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}


def format_cell(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def latex_escape(text):
    return "".join(LATEX_ESCAPES.get(char, char) for char in text)


def write_table(columns, rows, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cells = [[format_cell(row.get(column, "")) for column in columns] for row in rows]

    with open(directory / f"{stem}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(cells)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in cells:
        lines.append("| " + " | ".join(cell.replace("|", "/").replace("\n", " ") for cell in row) + " |")
    (directory / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    layout = "l" * len(columns)
    tex = [f"\\begin{{tabular}}{{{layout}}}", "\\hline"]
    tex.append(" & ".join(latex_escape(column) for column in columns) + " \\\\")
    tex.append("\\hline")
    for row in cells:
        tex.append(" & ".join(latex_escape(cell.replace("\n", " ")) for cell in row) + " \\\\")
    tex.extend(["\\hline", "\\end{tabular}"])
    (directory / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
