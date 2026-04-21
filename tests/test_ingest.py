"""
Tests for pipeline/ingest.py - PDFExtractor and text cleaning
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_fitz_span(text, font="helvetica", size=12.0, flags=0, origin=(10, 10)):
    span = {
        "text": text,
        "font": font,
        "size": size,
        "flags": flags,
        "origin": origin,
    }
    return span


def _make_fitz_block(lines_text, font="helvetica", size=12.0, flags=0, bbox=(0, 0, 500, 20)):
    lines = []
    for line_text in lines_text:
        span = _make_fitz_span(line_text, font=font, size=size, flags=flags)
        lines.append({"spans": [span]})
    return {"lines": lines, "bbox": bbox}


class TestBlockTypeTags:
    def test_page_number_detected(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            extractor = PDFExtractor.__new__(PDFExtractor)
            extractor.running_headers = {}
            extractor.pdf_path = Path("fake.pdf")
            extractor.doc = MagicMock()

        assert extractor._is_page_number("42", 42) is True
        assert extractor._is_page_number("100", 100) is True
        assert extractor._is_page_number("Page 5", 5) is True

    def test_page_number_false_for_body_text(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            extractor = PDFExtractor.__new__(PDFExtractor)
            extractor.running_headers = {}
            extractor.pdf_path = Path("fake.pdf")
            extractor.doc = MagicMock()

        assert extractor._is_page_number("This is a sentence.", 1) is False
        assert extractor._is_page_number("Introduction", 1) is False

    def test_figure_caption_detected(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            extractor = PDFExtractor.__new__(PDFExtractor)
            extractor.running_headers = {}
            extractor.doc = MagicMock()

        assert extractor._is_figure_caption("Figure 1: Attention weights") is True
        assert extractor._is_figure_caption("Fig. 3 — Memory layout") is True
        assert extractor._is_figure_caption("Not a figure") is False

    def test_footnote_detected(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            extractor = PDFExtractor.__new__(PDFExtractor)
            extractor.running_headers = {}
            extractor.doc = MagicMock()

        assert extractor._is_footnote("1 This is a short note") is True
        assert extractor._is_footnote("* Special note") is True

    def test_code_block_detected_by_monospace_font(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            extractor = PDFExtractor.__new__(PDFExtractor)
            extractor.running_headers = {}
            extractor.doc = MagicMock()

        assert extractor._is_code_block("x = 1", {"font": "Courier New"}) is True
        assert extractor._is_code_block("x = 1", {"font": "Consolas"}) is True

    def test_code_block_detected_by_pattern(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            extractor = PDFExtractor.__new__(PDFExtractor)
            extractor.running_headers = {}
            extractor.doc = MagicMock()

        assert extractor._is_code_block("def my_func():", {}) is True
        assert extractor._is_code_block("import numpy as np", {}) is True
        assert extractor._is_code_block("for i in range(10):", {}) is True

    def test_algorithm_detected(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            extractor = PDFExtractor.__new__(PDFExtractor)
            extractor.running_headers = {}
            extractor.doc = MagicMock()

        assert extractor._is_algorithm("Algorithm 1: Quick Sort") is True
        assert extractor._is_algorithm("Input: a sequence of values") is True
        assert extractor._is_algorithm("Output: sorted sequence") is True


class TestShouldKeepBlock:
    def setup_method(self):
        from pipeline.ingest import PDFExtractor, BlockType

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            self.extractor = PDFExtractor.__new__(PDFExtractor)
            self.extractor.running_headers = {}
            self.extractor.doc = MagicMock()
        self.BlockType = BlockType

    def test_page_number_dropped(self):
        assert self.extractor._should_keep_block(self.BlockType.PAGE_NUMBER, "42") is False

    def test_figure_caption_dropped(self):
        assert self.extractor._should_keep_block(self.BlockType.FIGURE_CAPTION, "Figure 1") is False

    def test_footnote_dropped(self):
        assert self.extractor._should_keep_block(self.BlockType.FOOTNOTE, "1 note") is False

    def test_running_header_dropped(self):
        assert self.extractor._should_keep_block(self.BlockType.RUNNING_HEADER, "Chapter 3") is False

    def test_code_block_kept(self):
        assert self.extractor._should_keep_block(self.BlockType.CODE, "x = 1") is True

    def test_algorithm_kept(self):
        assert self.extractor._should_keep_block(self.BlockType.PSEUDOCODE, "Algorithm 1") is True

    def test_body_with_content_kept(self):
        assert self.extractor._should_keep_block(self.BlockType.BODY, "Meaningful content") is True

    def test_body_empty_dropped(self):
        assert self.extractor._should_keep_block(self.BlockType.BODY, "   ") is False

    def test_table_with_content_kept(self):
        assert self.extractor._should_keep_block(self.BlockType.TABLE, "Column1  Column2  Column3") is True

    def test_table_too_short_dropped(self):
        assert self.extractor._should_keep_block(self.BlockType.TABLE, "AB") is False


class TestCleanText:
    def setup_method(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            self.extractor = PDFExtractor.__new__(PDFExtractor)
            self.extractor.running_headers = {}
            self.extractor.doc = MagicMock()

    def test_collapses_whitespace(self):
        result = self.extractor._clean_text("hello   world  test")
        assert result == "hello world test"

    def test_fixes_hyphenated_line_breaks(self):
        result = self.extractor._clean_text("con-\nnection")
        assert "connection" in result

    def test_normalizes_ligatures(self):
        result = self.extractor._clean_text("ﬁle ﬂow")
        assert "file" in result or "fi" in result

    def test_adds_math_flag_for_sum_symbol(self):
        result = self.extractor._clean_text("The formula is ∑ over all tokens")
        assert "[MATH]" in result

    def test_adds_math_flag_for_integral(self):
        result = self.extractor._clean_text("Compute ∫ f(x) dx")
        assert "[MATH]" in result

    def test_no_math_flag_for_plain_text(self):
        result = self.extractor._clean_text("Plain English text here")
        assert "[MATH]" not in result

    def test_strips_leading_trailing_whitespace(self):
        result = self.extractor._clean_text("  hello  ")
        assert result == "hello"


class TestTableToProse:
    def setup_method(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            self.extractor = PDFExtractor.__new__(PDFExtractor)
            self.extractor.running_headers = {}
            self.extractor.doc = MagicMock()

    def test_converts_table_to_prose(self):
        table = "Model  Latency  Throughput\ngpt-4  50ms  100 rps\nllama  20ms  200 rps"
        result = self.extractor._table_to_prose(table)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_line_returns_original(self):
        text = "Only one line"
        result = self.extractor._table_to_prose(text)
        assert result == text


class TestFinalCleanup:
    def setup_method(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open"), \
             patch.object(PDFExtractor, "_detect_running_headers"):
            self.extractor = PDFExtractor.__new__(PDFExtractor)
            self.extractor.running_headers = {}
            self.extractor.doc = MagicMock()

    def test_removes_excessive_blank_lines(self):
        text = "HEADER: Title\n\n\n\nBODY: Content"
        result = self.extractor._final_cleanup(text)
        assert "\n\n\n\n" not in result

    def test_result_is_stripped(self):
        result = self.extractor._final_cleanup("  content  ")
        assert result == result.strip()


class TestFileNotFound:
    def test_raises_when_pdf_missing(self):
        from pipeline.ingest import PDFExtractor
        with pytest.raises(FileNotFoundError):
            PDFExtractor("/nonexistent/path/book.pdf")


class TestContextManager:
    def test_context_manager_calls_close(self):
        from pipeline.ingest import PDFExtractor

        with patch("pipeline.ingest.fitz.open") as mock_open, \
             patch.object(PDFExtractor, "_detect_running_headers"):
            mock_doc = MagicMock()
            mock_open.return_value = mock_doc

            # Patch exists check
            with patch("pipeline.ingest.Path.exists", return_value=True):
                extractor = PDFExtractor("fake.pdf")
                extractor.__exit__(None, None, None)
                mock_doc.close.assert_called_once()
