"""Convert report.md to report.pdf using fpdf2."""
from fpdf import FPDF
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        fonts_dir = Path("C:/Windows/Fonts")
        self.add_font("Arial", "", str(fonts_dir / "arial.ttf"))
        self.add_font("Arial", "B", str(fonts_dir / "arialbd.ttf"))
        self.add_font("Arial", "I", str(fonts_dir / "ariali.ttf"))
        self.add_font("Consolas", "", str(fonts_dir / "consola.ttf"))

    def header(self):
        if self.page_no() > 1:
            self.set_font("Arial", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, "EEG-Based Emotion Classification: Three-Tier Evaluation", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def parse_table(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("|") and not re.match(r"^\|[\s\-:|]+\|$", line):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            rows.append(cells)
    return (rows[0], rows[1:]) if rows else ([], [])


def add_table(pdf, header, rows):
    if not header:
        return
    pdf.set_x(pdf.l_margin)
    n_cols = len(header)
    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    # Dynamic column widths
    all_rows = [header] + rows
    col_widths = []
    for col_i in range(n_cols):
        max_len = max((len(r[col_i]) if col_i < len(r) else 0) for r in all_rows)
        col_widths.append(max(max_len, 3))
    total = sum(col_widths)
    col_widths = [w / total * page_width for w in col_widths]

    # Header row
    pdf.set_font("Arial", "B", 7)
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(header):
        w = col_widths[i]
        pdf.cell(w, 6, h[:int(w / 1.6)] if len(h) > int(w / 1.6) else h, border=1, align="C", fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Arial", "", 7)
    pdf.set_text_color(0, 0, 0)
    for ri, row in enumerate(rows):
        pdf.set_x(pdf.l_margin)
        pdf.set_fill_color(248, 249, 250) if ri % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        for i in range(n_cols):
            w = col_widths[i]
            txt = row[i] if i < len(row) else ""
            max_c = int(w / 1.6)
            if len(txt) > max_c:
                txt = txt[:max_c - 1] + "\u2026"
            pdf.cell(w, 5.5, txt, border=1, align="C", fill=True)
        pdf.ln()
    pdf.ln(3)


def clean_text(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', text)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r'\$\$([^\$]+)\$\$', r'\1', text)
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)
    # Keep single * for now, remove them
    text = text.replace("*", "")
    return text


def build_pdf():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    md_text = (BASE_DIR / "report.md").read_text(encoding="utf-8")
    lines = md_text.split("\n")

    i = 0
    in_code = False
    code_lines = []
    table_lines = []
    in_table = False

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            if in_code:
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Consolas", "", 7)
                pdf.set_fill_color(245, 245, 245)
                for cl in code_lines:
                    pdf.cell(0, 4, cl, new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.ln(3)
                code_lines = []
                in_code = False
            else:
                if in_table:
                    add_table(pdf, *parse_table(table_lines))
                    table_lines = []
                    in_table = False
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Tables
        if line.strip().startswith("|"):
            in_table = True
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            add_table(pdf, *parse_table(table_lines))
            table_lines = []
            in_table = False

        # H1
        if line.startswith("# ") and not line.startswith("## "):
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Arial", "B", 18)
            pdf.set_text_color(44, 62, 80)
            pdf.multi_cell(0, 10, clean_text(line[2:].strip()), align="C")
            pdf.ln(3)
            pdf.set_draw_color(44, 62, 80)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(6)
            i += 1
            continue

        # H2
        if line.startswith("## "):
            pdf.set_x(pdf.l_margin)
            pdf.ln(5)
            pdf.set_font("Arial", "B", 13)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 8, clean_text(line[3:].strip()), new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(189, 195, 199)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # H3
        if line.startswith("### "):
            pdf.set_x(pdf.l_margin)
            pdf.ln(3)
            pdf.set_font("Arial", "B", 11)
            pdf.set_text_color(52, 73, 94)
            pdf.cell(0, 7, clean_text(line[4:].strip()), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            i += 1
            continue

        # HR
        if line.strip() == "---":
            pdf.ln(3)
            pdf.set_draw_color(189, 195, 199)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # Images
        img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if img_match:
            img_path = BASE_DIR / img_match.group(2)
            if img_path.exists():
                pdf.set_x(pdf.l_margin)
                pdf.ln(3)
                w = pdf.w - pdf.l_margin - pdf.r_margin - 10
                pdf.image(str(img_path), x=pdf.l_margin + 5, w=w)
                pdf.ln(3)
                pdf.set_x(pdf.l_margin)
            else:
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Arial", "I", 9)
                pdf.set_text_color(128, 128, 128)
                pdf.cell(0, 5, f"[Image: {img_match.group(1)}]", new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue

        # Blank
        if line.strip() == "":
            pdf.ln(2)
            i += 1
            continue

        # Regular text
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(26, 26, 26)
        text = clean_text(line.strip())

        if text.startswith("- ") or text.startswith("* "):
            text = "  \u2022 " + text[2:]
        elif re.match(r"^\d+\. ", text):
            m = re.match(r"^(\d+)\. (.+)", text)
            if m:
                text = f"  {m.group(1)}. {m.group(2)}"

        if text:
            pdf.multi_cell(0, 5, text)

        i += 1

    if in_table:
        add_table(pdf, *parse_table(table_lines))

    output_path = BASE_DIR / "report.pdf"
    pdf.output(str(output_path))
    print(f"PDF generated: {output_path}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_pdf()
