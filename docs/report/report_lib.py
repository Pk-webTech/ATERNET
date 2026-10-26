"""
Small python-docx helper layer used to build the project report on top of the college template.

It keeps one running state (figure / table numbers, bookmarks, citations) so that the chapter text
only has to say what goes where.
"""

import copy
import re
from pathlib import Path

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
TEXT_WIDTH_CM = 15.6          # A4 width minus the template's left/right margins
J, C_, L_, R_ = (WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER,
                 WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT)


def set_font(run, size=12, bold=None, italic=None, name=FONT, color=None, highlight=False):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(a), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if highlight:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def roman(n):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


class Report:
    def __init__(self, doc, pages=None):
        self.doc = doc
        self.pages = pages or {}          # bookmark -> page text (from a previous Word pagination pass)
        self.bm_id = 500
        self.fig_count, self.tab_count = {}, {}
        self.figs, self.tabs, self.toc = [], [], []   # (label, title, bookmark)
        self.cites = {}                    # key -> number (order of first citation)
        self.ref_text = {}                 # key -> IEEE string
        self.chapter = 0
        self.bookmarks = []

    # ------------------------------------------------------------------ low level
    def _bookmark(self, p, name):
        self.bm_id += 1
        start = OxmlElement("w:bookmarkStart")
        start.set(qn("w:id"), str(self.bm_id))
        start.set(qn("w:name"), name)
        end = OxmlElement("w:bookmarkEnd")
        end.set(qn("w:id"), str(self.bm_id))
        ppr = p._p.find(qn("w:pPr"))
        p._p.insert(1 if ppr is not None else 0, start)
        p._p.append(end)
        self.bookmarks.append(name)

    def page_text(self, bm):
        return self.pages.get(bm, "0")

    def pageref(self, paragraph, bm, size=12, bold=False):
        """Insert a PAGEREF field (cached with the last known page) into a paragraph."""
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), f" PAGEREF {bm} \\h ")
        run = paragraph.add_run(self.page_text(bm))
        set_font(run, size, bold)
        fld.append(run._r)
        paragraph._p.append(fld)

    # ------------------------------------------------------------------ paragraphs
    def para(self, text="", align=J, size=12, bold=False, italic=False, before=0, after=6, spacing=1.15,
             first_indent=None, left_indent=None, keep_next=False, page_break=False, style=None):
        p = self.doc.add_paragraph(style=style)
        pf = p.paragraph_format
        p.alignment = align
        pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), spacing
        if first_indent is not None:
            pf.first_line_indent = Cm(first_indent)
        if left_indent is not None:
            pf.left_indent = Cm(left_indent)
        pf.keep_with_next = keep_next
        pf.page_break_before = page_break
        if text:
            self.add_text(p, text, size, bold, italic)
        return p

    def add_text(self, p, text, size=12, bold=False, italic=False):
        """Supports **bold** and _italic_ inline markers and [@key] citations."""
        text = re.sub(r"\[@([a-z0-9_,]+)\]", lambda m: self.cite(*m.group(1).split(",")), text)
        for part in re.split(r"(\*\*[^*]+\*\*|__[^_]+__)", text):
            if not part:
                continue
            if part.startswith("**"):
                r = p.add_run(part[2:-2]); set_font(r, size, True, italic)
            elif part.startswith("__"):
                r = p.add_run(part[2:-2]); set_font(r, size, bold, True)
            else:
                r = p.add_run(part); set_font(r, size, bold, italic)

    def body(self, text, **kw):
        kw.setdefault("first_indent", 0.0)
        return self.para(text, **kw)

    def bullets(self, items, size=12, numbered=False, left=1.0):
        for k, it in enumerate(items, 1):
            p = self.para("", size=size, after=3, left_indent=left, first_indent=-0.6)
            lead = f"{k}." if numbered else "•"
            r = p.add_run(lead + "\t"); set_font(r, size)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(left))
            self.add_text(p, it, size)

    # ------------------------------------------------------------------ headings
    def chapter_heading(self, num, title):
        self.chapter = num
        self.fig_count[num], self.tab_count[num] = 0, 0
        p = self.para(f"CHAPTER {num}", align=C_, size=16, bold=True, before=0, after=2, spacing=1.0,
                      page_break=True, keep_next=True)
        q = self.para(title.upper(), align=C_, size=14, bold=True, before=0, after=10, spacing=1.0, keep_next=True)
        bm = f"h_{num}"
        self._bookmark(q, bm)
        self.toc.append((0, f"{num}.", title.upper(), bm))
        return q

    def h2(self, num, title, toc=True):
        p = self.para(f"{num} {title}", align=L_, size=14, bold=True, before=10, after=6, spacing=1.0, keep_next=True)
        bm = "h_" + num.replace(".", "_")
        self._bookmark(p, bm)
        if toc:
            self.toc.append((1, "", f"{num} {title}", bm))
        return p

    def h3(self, num, title, toc=True):
        p = self.para(f"{num} {title}", align=L_, size=12, bold=True, before=8, after=4, spacing=1.0, keep_next=True)
        bm = "h_" + num.replace(".", "_")
        self._bookmark(p, bm)
        if toc:
            self.toc.append((2, "", f"{num} {title}", bm))
        return p

    def plain_heading(self, title, toc_title=None, level=0, page_break=True, size=16):
        """Heading for REFERENCES / APPENDIX (no chapter number)."""
        p = self.para(title, align=C_, size=size, bold=True, before=0, after=10, spacing=1.0,
                      page_break=page_break, keep_next=True)
        bm = "h_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        self._bookmark(p, bm)
        self.toc.append((level, "", toc_title or title.upper(), bm))
        return p

    # ------------------------------------------------------------------ figures and tables
    def fig(self, path, caption, width_cm=14.8, max_h_cm=19.0, key=None):
        ch = self.chapter
        self.fig_count[ch] = self.fig_count.get(ch, 0) + 1
        label = f"Figure {ch}.{self.fig_count[ch]}"
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        width = min(width_cm, max_h_cm * w / h)
        p = self.para("", align=C_, before=4, after=2, spacing=1.0, keep_next=True)
        p.add_run().add_picture(str(path), width=Cm(width))
        cap = self.para("", align=C_, size=10, before=0, after=10, spacing=1.0)
        r = cap.add_run(f"{label}: "); set_font(r, 10, True)
        self.add_text(cap, caption, 10)
        bm = f"fig_{ch}_{self.fig_count[ch]}"
        self._bookmark(cap, bm)
        self.figs.append((f"{ch}.{self.fig_count[ch]}", caption, bm))
        return label

    def next_fig_label(self):
        return f"Figure {self.chapter}.{self.fig_count.get(self.chapter, 0) + 1}"

    def next_tab_label(self):
        return f"Table {self.chapter}.{self.tab_count.get(self.chapter, 0) + 1}"

    def table(self, header, rows, caption, widths=None, size=10, align=None, bold_first_col=False, highlight_rows=()):
        ch = self.chapter
        self.tab_count[ch] = self.tab_count.get(ch, 0) + 1
        label = f"Table {ch}.{self.tab_count[ch]}"
        cap = self.para("", align=C_, size=10, before=6, after=3, spacing=1.0, keep_next=True)
        r = cap.add_run(f"{label}: "); set_font(r, 10, True)
        self.add_text(cap, caption, 10)
        bm = f"tab_{ch}_{self.tab_count[ch]}"
        self._bookmark(cap, bm)
        self.tabs.append((f"{ch}.{self.tab_count[ch]}", caption, bm))

        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        widths = widths or [TEXT_WIDTH_CM / len(header)] * len(header)
        for i, htxt in enumerate(header):
            self._cell(t.rows[0].cells[i], htxt, size, True, shade="E7E6E6", align=C_, width=widths[i])
        trpr = t.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader"); hdr.set(qn("w:val"), "true"); trpr.append(hdr)
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                a = (align[i] if align else (L_ if i == 0 else C_))
                self._cell(cells[i], str(v), size, bold_first_col and i == 0, align=a, width=widths[i],
                           shade="DDEBF7" if ri in highlight_rows else None)
        for row in t.rows:
            trpr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trpr.append(cs)
        self.para("", after=6, spacing=1.0)
        return label

    def _cell(self, cell, text, size, bold, shade=None, align=L_, width=None):
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = align
        p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        self.add_text(p, text, size, bold)
        if width:
            cell.width = Cm(width)
        if shade:
            tcpr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), shade)
            tcpr.append(shd)

    def equation(self, expr, number=None):
        ch = self.chapter
        self.fig_count.setdefault("eq" + str(ch), 0)
        self.fig_count["eq" + str(ch)] += 1
        n = f"({ch}.{self.fig_count['eq' + str(ch)]})"
        p = self.para("", align=L_, before=4, after=6, spacing=1.15, keep_next=False)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM / 2), alignment=1)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), alignment=2)
        r = p.add_run("\t" + expr); set_font(r, 12, False, True)
        r = p.add_run("\t" + n); set_font(r, 12)
        return n

    def code(self, text, size=7.0):
        for line in text.rstrip("\n").split("\n"):
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_before = pf.space_after = Pt(0)
            pf.line_spacing = 1.0
            pf.left_indent = Cm(0.3)
            r = p.add_run(line if line else " ")
            set_font(r, size, name="Courier New")
            ppr = p._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F2F2")
            ppr.append(shd)

    # ------------------------------------------------------------------ citations
    def cite(self, *keys):
        nums = []
        for k in keys:
            if k not in self.ref_text:
                raise KeyError(f"unknown reference key {k}")
            if k not in self.cites:
                self.cites[k] = len(self.cites) + 1
            nums.append(self.cites[k])
        nums.sort()
        return "[" + ", ".join(str(n) for n in nums) + "]"
