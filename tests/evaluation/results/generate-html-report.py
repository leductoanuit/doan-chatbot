"""Generate HTML report from eval markdown report."""
import re
import sys
from pathlib import Path

MD_FILE = Path(__file__).parent / "bao-cao-ket-qua-eval.md"
OUT_FILE = Path(__file__).parent / "bao-cao-ket-qua-eval.html"

md = MD_FILE.read_text(encoding="utf-8")

def md_to_html(text):
    lines = text.split("\n")
    html_lines = []
    in_table = False
    in_blockquote = False
    in_list = False
    table_rows = []

    def flush_table():
        if not table_rows:
            return ""
        result = ['<table>']
        for i, row in enumerate(table_rows):
            cells = [c.strip() for c in row.strip("|").split("|")]
            if i == 0:
                result.append("<thead><tr>" + "".join(f"<th>{fmt(c)}</th>" for c in cells) + "</tr></thead><tbody>")
            elif i == 1 and all(re.match(r"^[-:]+$", c.strip()) for c in cells):
                continue
            else:
                result.append("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in cells) + "</tr>")
        result.append("</tbody></table>")
        return "\n".join(result)

    def fmt(s):
        # bold
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        # inline code
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    i = 0
    while i < len(lines):
        line = lines[i]

        # Table detection
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(line)
            i += 1
            continue
        else:
            if in_table:
                html_lines.append(flush_table())
                table_rows = []
                in_table = False

        # HR
        if re.match(r"^---+$", line.strip()):
            html_lines.append("<hr>")
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            html_lines.append(f"<h{level}>{fmt(m.group(2))}</h{level}>")
            i += 1
            continue

        # Blockquote
        if line.startswith("> "):
            html_lines.append(f'<blockquote>{fmt(line[2:])}</blockquote>')
            i += 1
            continue

        # List item
        if re.match(r"^(\d+\.|[-*])\s+", line):
            tag = "ol" if re.match(r"^\d+\.", line) else "ul"
            if not in_list:
                html_lines.append(f"<{tag}>")
                in_list = tag
            content = re.sub(r"^(\d+\.|[-*])\s+", "", line)
            html_lines.append(f"<li>{fmt(content)}</li>")
            i += 1
            # check next line
            if i < len(lines) and not re.match(r"^(\d+\.|[-*])\s+", lines[i]):
                html_lines.append(f"</{in_list}>")
                in_list = False
            continue

        # Empty line
        if not line.strip():
            html_lines.append("")
            i += 1
            continue

        # Paragraph
        html_lines.append(f"<p>{fmt(line)}</p>")
        i += 1

    if in_table:
        html_lines.append(flush_table())

    return "\n".join(html_lines)

body = md_to_html(md)

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo Cáo Kết Quả Đánh Giá RAG Chatbot UIT</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 14px;
      line-height: 1.7;
      color: #2c3e50;
      background: #f5f7fa;
      padding: 40px 20px;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 20px rgba(0,0,0,0.08);
      padding: 50px 60px;
    }}
    h1 {{
      font-size: 26px;
      color: #1a252f;
      border-bottom: 3px solid #3498db;
      padding-bottom: 12px;
      margin-bottom: 20px;
    }}
    h2 {{
      font-size: 20px;
      color: #2980b9;
      margin: 32px 0 12px;
      padding-left: 10px;
      border-left: 4px solid #3498db;
    }}
    h3 {{
      font-size: 16px;
      color: #2c3e50;
      margin: 20px 0 8px;
    }}
    p {{ margin: 8px 0; }}
    hr {{
      border: none;
      border-top: 1px solid #ecf0f1;
      margin: 28px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 13px;
    }}
    thead tr {{ background: #2980b9; color: #fff; }}
    th, td {{
      padding: 10px 14px;
      text-align: left;
      border: 1px solid #dde;
    }}
    tbody tr:nth-child(even) {{ background: #f4f8fb; }}
    tbody tr:hover {{ background: #e8f4fd; }}
    blockquote {{
      background: #fff8e1;
      border-left: 4px solid #f39c12;
      padding: 10px 16px;
      margin: 12px 0;
      border-radius: 4px;
      color: #7f6000;
    }}
    code {{
      background: #ecf0f1;
      padding: 2px 6px;
      border-radius: 3px;
      font-family: monospace;
      font-size: 12px;
      color: #c0392b;
    }}
    strong {{ color: #1a252f; }}
    ul, ol {{
      margin: 8px 0 8px 24px;
    }}
    li {{ margin: 4px 0; }}
    .footer {{
      margin-top: 40px;
      padding-top: 16px;
      border-top: 1px solid #ecf0f1;
      font-size: 12px;
      color: #95a5a6;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="container">
    {body}
    <div class="footer">
      Được tạo tự động từ eval-summary JSON files · UIT RAG Chatbot Evaluation
    </div>
  </div>
</body>
</html>"""

OUT_FILE.write_text(html, encoding="utf-8")
print(f"HTML report saved: {OUT_FILE}")
