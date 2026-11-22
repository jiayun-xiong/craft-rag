import tempfile
import unittest
from pathlib import Path
import pymupdf
from src.parsing.parser import PdfParser, joined_tables


class ParsingTests(unittest.TestCase):
    def test_two_page_grid_keeps_source_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pdf"
            with pymupdf.open() as doc:
                for y, title, value in [(165, "Tools", "A"), (30, "Tools continued", "B")]:
                    page = doc.new_page(width=200, height=220)
                    page.insert_text((20, y-5), title)
                    for x in [20, 90, 160]:
                        page.draw_line((x, y), (x, y+40))
                    for h in [0, 20, 40]:
                        page.draw_line((20, y+h), (160, y+h))
                    for x, dy, label in [(25, 15, "Tool"), (95, 15, "Unit"), (25, 35, value), (95, 35, "mm")]:
                        page.insert_text((x, y+dy), label)
                doc.save(path)
            rows = joined_tables(PdfParser().parse(path))["p1-t1"]
            self.assertEqual([(r["page"], r["row"][0]) for r in rows], [(1,"Tool"), (1,"A"), (2,"B")])

    def test_catalogue_open_edge_and_two_tables(self):
        path = Path(__file__).resolve().parents[1] / "data/raw_pdfs/MMC-B196C.pdf"
        if not path.exists():
            self.skipTest("需要本地三菱原件")
        upper, lower = PdfParser().parse(path, [19])[0]["tables"]
        self.assertEqual(upper["rows"][1][-2:], ["0.08─0.25", "0.10─1.00"])
        self.assertEqual(upper["rows"][2][2], upper["rows"][1][2])
        self.assertIn("正角", lower["preceding_text"])
        self.assertEqual(lower["rows"][1][-1], "0.20─0.90")
