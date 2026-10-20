"""Guardrail 1: visible_pdf_text drops hidden text and counts it."""
import tools
from conftest import CLEAN_LINES, WHITE, nonspace
from pdf_factory import make_pdf


def test_clean_pdf_has_no_hidden_text(pdfs):
    text, report = tools.visible_pdf_text(str(pdfs["clean"]))
    for line in CLEAN_LINES:
        assert line in text
    assert report == {"hidden_chars": 0, "hidden_sample": "", "reasons": {}}


def test_white_text_is_dropped(pdfs):
    text, report = tools.visible_pdf_text(str(pdfs["white"]))
    assert "SYSTEM NOTE" not in text and "Spark" not in text
    assert "Built dashboards and data pipelines in Python and SQL" in text  # visible text kept
    assert report["hidden_chars"] == sum(nonspace(t) for t in WHITE)
    assert report["reasons"] == {"white": report["hidden_chars"]}
    assert report["hidden_sample"].startswith("Python SQL Spark")
    assert len(report["hidden_sample"]) <= 80


def test_tiny_offpage_and_invisible_text_are_dropped(pdfs):
    cases = {"tiny": ("Kubernetes", "tiny"), "offpage": ("15 years", "offpage"),
             "invisible": ("Ignore previous", "invisible")}
    for name, (secret, reason) in cases.items():
        text, report = tools.visible_pdf_text(str(pdfs[name]))
        assert secret not in text, name
        assert "Jordan Sample" in text, name
        assert report["reasons"].get(reason, 0) == report["hidden_chars"] > 0, name


def test_white_text_on_a_dark_box_is_visible(tmp_path):
    # CV templates often have white text on a coloured header: a reader sees it, so keep it.
    f = make_pdf(tmp_path / "header.pdf", hidden=[("Senior Data Analyst", "white_on_dark")])
    text, report = tools.visible_pdf_text(str(f))
    assert "Senior Data Analyst" in text and report["hidden_chars"] == 0
    # ...but a white box drawn over the dark one makes it hidden again
    f = make_pdf(tmp_path / "trick.pdf", hidden=[("Senior Data Analyst", "white_on_white_box")])
    text, report = tools.visible_pdf_text(str(f))
    assert "Senior Data Analyst" not in text and report["reasons"] == {"white": nonspace("Senior Data Analyst")}


def test_colour_spaces():
    assert tools._is_white((1.0,)) and tools._is_white(0.97)                 # gray
    assert tools._is_white((1.0, 0.96, 0.99))                                 # RGB near-white
    assert tools._is_white((0, 0, 0, 0))                                      # CMYK white
    assert not tools._is_white((0, 0, 0, 1))                                  # CMYK black
    assert not tools._is_white((0.9, 1, 1)) and not tools._is_white(None)     # light blue / default
    assert not tools._is_white(("pattern",))


def test_cache_and_bad_files(pdfs, tmp_path):
    first = tools.visible_pdf_text(str(pdfs["white"]))
    assert tools.visible_pdf_text(str(pdfs["white"])) == first               # cached
    assert tools.visible_pdf_text(str(tmp_path / "missing.pdf"))[0] == ""
    assert tools.visible_pdf_text(None)[1]["hidden_chars"] == 0


def test_scanned_pdf_falls_back_to_old_sources(pdfs):
    assert not tools.has_text_layer(str(pdfs["scanned"]))
    assert tools.quote_sources(str(pdfs["scanned"]), "parser text")[0] == "parser text"
    # with a text layer, the parser's text is NOT a source any more
    sources = tools.quote_sources(str(pdfs["white"]), "parser text with SYSTEM NOTE")
    assert all("SYSTEM NOTE" not in s for s in sources)
