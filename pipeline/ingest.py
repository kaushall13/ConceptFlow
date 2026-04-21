"""
PDF Ingestion & Cleaning - Extract and clean technical PDF content with structure tagging
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF for PDF processing


class BlockType:
    """Block type constants for tagging."""
    HEADER = "HEADER"
    SUBHEADER = "SUBHEADER"
    BODY = "BODY"
    CODE = "CODE"
    PSEUDOCODE = "ALGORITHM"  # Algorithms treated as code
    TABLE = "TABLE"
    FIGURE_CAPTION = "FIGURE-CAPTION"
    FOOTNOTE = "FOOTNOTE"
    PAGE_NUMBER = "PAGE-NUMBER"
    RUNNING_HEADER = "RUNNING-HEADER"


class PDFExtractor:
    """Extracts and cleans PDF content with structure tagging."""

    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Open PDF with PyMuPDF
        self.doc = fitz.open(self.pdf_path)

        # Track running headers to filter them out
        self.running_headers: Dict[int, str] = {}
        self._detect_running_headers()

    def _detect_running_headers(self):
        """Detect running headers across pages to filter them out."""
        # Simple heuristic: text that appears at same position on multiple consecutive pages
        header_candidates = {}

        for page_num in range(min(10, len(self.doc))):  # Check first 10 pages
            page = self.doc[page_num]
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text and len(text) < 50:  # Running headers are short
                                # Use position as key
                                pos_key = (int(span["origin"][0]), int(span["origin"][1]))
                                if pos_key not in header_candidates:
                                    header_candidates[pos_key] = {"text": text, "count": 0}
                                header_candidates[pos_key]["count"] += 1

        # Identify headers that appear on 3+ pages at same position
        for pos_key, data in header_candidates.items():
            if data["count"] >= 3:
                # Store normalized version
                normalized = self._normalize_text(data["text"])
                self.running_headers[pos_key] = normalized

    def extract_and_clean(self) -> str:
        """
        Extract and clean PDF content with block type tagging.

        Returns:
            Clean tagged text string
        """
        tagged_blocks = []

        for page_num, page in enumerate(self.doc, 1):
            print(f"  Processing page {page_num}/{len(self.doc)}")

            page_blocks = self._process_page(page, page_num)
            tagged_blocks.extend(page_blocks)

        # Join blocks with newlines and apply final cleanup
        clean_text = "\n".join(tagged_blocks)
        clean_text = self._final_cleanup(clean_text)

        return clean_text

    def _process_page(self, page, page_num: int) -> List[str]:
        """
        Process a single page and return tagged blocks.

        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)

        Returns:
            List of tagged block strings
        """
        tagged_blocks = []

        # Check if page has multiple columns
        is_multi_column = self._detect_multi_column(page)

        if is_multi_column:
            blocks = self._extract_multi_column(page)
        else:
            blocks = self._extract_single_column(page)

        for block in blocks:
            block_type, content = self._classify_block(block, page_num)

            if self._should_keep_block(block_type, content):
                tagged_content = self._clean_block_content(block_type, content)
                tagged_blocks.append(f"{block_type}: {tagged_content}")

        return tagged_blocks

    def _detect_multi_column(self, page) -> bool:
        """
        Detect if page has multiple columns.

        Args:
            page: PyMuPDF page object

        Returns:
            True if multi-column detected
        """
        # Simple heuristic: check if text blocks are distributed across page width
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        if len(blocks) < 3:
            return False

        # Get x-positions of all blocks
        x_positions = [block["bbox"][0] for block in blocks if "lines" in block]

        if not x_positions:
            return False

        # If blocks are spread across page, likely multi-column
        page_width = page.rect.width
        x_range = max(x_positions) - min(x_positions)

        return x_range > page_width * 0.4  # Spread across >40% of page width

    def _extract_single_column(self, page) -> List[Dict[str, Any]]:
        """
        Extract blocks from single-column page.

        Args:
            page: PyMuPDF page object

        Returns:
            List of block dictionaries with text and metadata
        """
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        extracted_blocks = []
        for block in blocks:
            if "lines" not in block:
                continue

            text_lines = []
            for line in block["lines"]:
                line_text = " ".join(span["text"] for span in line["spans"])
                text_lines.append(line_text)

            if text_lines:
                extracted_blocks.append({
                    "text": "\n".join(text_lines),
                    "bbox": block["bbox"],
                    "font_info": self._get_font_info(block)
                })

        return extracted_blocks

    def _extract_multi_column(self, page) -> List[Dict[str, Any]]:
        """
        Extract and reorder blocks from multi-column page.

        Args:
            page: PyMuPDF page object

        Returns:
            List of block dictionaries reordered left-to-right
        """
        blocks = self._extract_single_column(page)

        # Group blocks by column (x-coordinate)
        page_width = page.rect.width
        mid_point = page_width / 2

        left_column = []
        right_column = []

        for block in blocks:
            x_center = (block["bbox"][0] + block["bbox"][2]) / 2
            if x_center < mid_point:
                left_column.append(block)
            else:
                right_column.append(block)

        # Sort each column by y-coordinate (top to bottom)
        left_column.sort(key=lambda b: b["bbox"][1])
        right_column.sort(key=lambda b: b["bbox"][1])

        # Interleave columns: read left-to-right, top-to-bottom
        reordered = []
        max_blocks = max(len(left_column), len(right_column))

        for i in range(max_blocks):
            if i < len(left_column):
                reordered.append(left_column[i])
            if i < len(right_column):
                reordered.append(right_column[i])

        return reordered

    def _classify_block(self, block: Dict[str, Any], page_num: int) -> Tuple[str, str]:
        """
        Classify a block's type and extract content.

        Args:
            block: Block dictionary with text and metadata
            page_num: Current page number

        Returns:
            Tuple of (block_type, content)
        """
        text = block["text"].strip()
        font_info = block.get("font_info", {})

        if not text:
            return BlockType.BODY, ""

        # Check for page numbers
        if self._is_page_number(text, page_num):
            return BlockType.PAGE_NUMBER, ""

        # Check for running headers
        if self._is_running_header(block, text):
            return BlockType.RUNNING_HEADER, ""

        # Check for figure captions
        if self._is_figure_caption(text):
            return BlockType.FIGURE_CAPTION, ""

        # Check for footnotes
        if self._is_footnote(text):
            return BlockType.FOOTNOTE, ""

        # Check for code blocks
        if self._is_code_block(text, font_info):
            return BlockType.CODE, text

        # Check for algorithm/pseudocode
        if self._is_algorithm(text):
            return BlockType.PSEUDOCODE, text

        # Check for tables
        if self._is_table(text):
            return BlockType.TABLE, text

        # Check for headers
        font_size = font_info.get("size", 12)
        is_bold = font_info.get("flags", 0) & 2**4  # Bold flag in PyMuPDF

        if font_size > 16 and is_bold:
            return BlockType.HEADER, text
        elif font_size > 13 and is_bold:
            return BlockType.SUBHEADER, text

        # Default to body text
        return BlockType.BODY, text

    def _get_font_info(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract font information from block.

        Args:
            block: Block dictionary

        Returns:
            Dictionary with font size, flags, name
        """
        if "lines" not in block or not block["lines"]:
            return {}

        first_line = block["lines"][0]
        if not first_line.get("spans"):
            return {}

        first_span = first_line["spans"][0]
        return {
            "size": first_span.get("size", 12),
            "flags": first_span.get("flags", 0),
            "font": first_span.get("font", "")
        }

    def _is_page_number(self, text: str, page_num: int) -> bool:
        """Check if text is a page number."""
        # Check if text matches page number or is very short and numeric
        text_stripped = text.strip()

        # Direct match
        if str(page_num) == text_stripped:
            return True

        # Short numeric pattern
        if re.match(r'^\d{1,4}$', text_stripped):
            return True

        # Page pattern like "Page 42" or "p. 42"
        if re.match(r'^(page|p\.?)\s*\d+$', text_stripped.lower()):
            return True

        return False

    def _is_running_header(self, block: Dict[str, Any], text: str) -> bool:
        """Check if block is a running header."""
        pos_key = (int(block["bbox"][0]), int(block["bbox"][1]))

        # Check if position matches known running headers
        for header_pos, header_text in self.running_headers.items():
            # Allow some position tolerance
            if (abs(pos_key[0] - header_pos[0]) < 20 and
                abs(pos_key[1] - header_pos[1]) < 20):

                # Check if text is similar
                normalized = self._normalize_text(text)
                if normalized == header_text:
                    return True

        return False

    def _is_figure_caption(self, text: str) -> bool:
        """Check if text is a figure caption."""
        # Figure captions often start with "Figure", "Fig.", "Exhibit", etc.
        patterns = [
            r'^figure\s+\d+',
            r'^fig\.?\s+\d+',
            r'^exhibit\s+\d+',
            r'^diagram\s+\d+',
            r'^chart\s+\d+'
        ]

        text_lower = text.lower()
        return any(re.match(pattern, text_lower) for pattern in patterns)

    def _is_footnote(self, text: str) -> bool:
        """Check if text is a footnote."""
        # Footnotes often start with numbers or symbols
        if re.match(r'^\s*[\d\*\†\‡\§]\s*', text):
            return True

        # Check if text is at bottom of page (would need page context)
        # For now, use simple heuristic: short text starting with number
        if len(text) < 100 and re.match(r'^\s*\d+\s+', text):
            return True

        return False

    def _is_code_block(self, text: str, font_info: Dict[str, Any]) -> bool:
        """Check if block is a code block."""
        # Code blocks often use monospace fonts
        font_name = font_info.get("font", "").lower()
        monospace_fonts = ["courier", "mono", "code", "consolas", "monaco"]

        if any(monofont in font_name for monofont in monospace_fonts):
            return True

        # Check for code-like patterns
        code_indicators = [
            r'^\s*(def |function |class |import |public |private |protected )',
            r'^\s*(if |for |while |switch |case |return )',
            r'^\s*(\/\/|#|/\*|\*)',  # Comments
            r'[{}();,]\s*$',  # Lines ending with code symbols
            r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*',  # Assignments
        ]

        if any(re.search(pattern, text) for pattern in code_indicators):
            return True

        return False

    def _is_algorithm(self, text: str) -> bool:
        """Check if block is an algorithm/pseudocode."""
        # Algorithms often have specific markers
        patterns = [
            r'^algorithm\s+\d+',
            r'^procedure\s+\w+',
            r'^function\s+\w+\s*\(',
            r'^input:',
            r'^output:',
            r'^\d+:\s+',  # Numbered steps
        ]

        text_lower = text.lower()
        return any(re.match(pattern, text_lower) for pattern in patterns)

    def _is_table(self, text: str) -> bool:
        """Check if block contains table data."""
        # Tables have multiple columns with consistent spacing
        lines = text.split('\n')

        if len(lines) < 2:
            return False

        # Check for tab-like spacing or multiple columns
        for line in lines:
            # Count tab characters or multiple spaces
            if '\t' in line or '  ' in line:
                # Check if there are multiple "columns" (separated by tabs or spaces)
                parts = re.split(r'\t|  +', line.strip())
                if len(parts) >= 3:  # At least 3 columns
                    return True

        return False

    def _should_keep_block(self, block_type: str, content: str) -> bool:
        """
        Determine if a block should be kept in the output.

        Args:
            block_type: Type of the block
            content: Block content

        Returns:
            True if block should be kept
        """
        # Drop these entirely
        if block_type in [BlockType.PAGE_NUMBER, BlockType.RUNNING_HEADER,
                          BlockType.FIGURE_CAPTION, BlockType.FOOTNOTE]:
            return False

        # Keep code and algorithms exactly as-is
        if block_type in [BlockType.CODE, BlockType.PSEUDOCODE]:
            return True

        # For tables, we'll reconstruct as prose
        if block_type == BlockType.TABLE:
            return len(content.strip()) > 10  # Only keep non-empty tables

        # For headers and body, keep if there's content
        if block_type in [BlockType.HEADER, BlockType.SUBHEADER, BlockType.BODY]:
            return len(content.strip()) > 0

        return False

    def _clean_block_content(self, block_type: str, content: str) -> str:
        """
        Clean block content based on type.

        Args:
            block_type: Type of the block
            content: Raw content

        Returns:
            Cleaned content
        """
        # Code blocks: preserve exactly
        if block_type in [BlockType.CODE, BlockType.PSEUDOCODE]:
            return content

        # Tables: convert to prose
        if block_type == BlockType.TABLE:
            return self._table_to_prose(content)

        # Headers and body: apply text cleaning
        return self._clean_text(content)

    def _table_to_prose(self, table_text: str) -> str:
        """
        Convert table text to prose description.

        Args:
            table_text: Raw table text

        Returns:
            Prose description of table content
        """
        lines = table_text.split('\n')

        if len(lines) < 2:
            return table_text

        # Try to parse table structure
        # Simple approach: assume first line is header
        headers = re.split(r'\t|  +', lines[0].strip())
        data_rows = []

        for line in lines[1:]:
            if line.strip():
                cells = re.split(r'\t|  +', line.strip())
                if len(cells) == len(headers):
                    data_rows.append(cells)

        # Generate prose: "For X, Y is Z"
        prose_parts = []

        if data_rows:
            # Find main column (usually first or second)
            main_col = 0
            if len(headers) > 1 and 'name' in headers[0].lower():
                main_col = 1

            for row in data_rows:
                main_value = row[main_col]
                for i, (header, value) in enumerate(zip(headers, row)):
                    if i != main_col and value.strip():
                        prose_parts.append(f"For {main_value}, {header.lower()} is {value}.")

            return " ".join(prose_parts)

        return table_text  # Fallback to original

    def _clean_text(self, text: str) -> str:
        """
        Apply text cleaning operations.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix hyphenated line breaks
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

        # Normalize unicode ligatures
        text = unicodedata.normalize('NFKC', text)

        # Approximate garbled math with ASCII + [MATH] flag
        text = self._handle_math(text)

        return text.strip()

    def _handle_math(self, text: str) -> str:
        """
        Handle mathematical notation in text.

        Args:
            text: Text possibly containing math

        Returns:
            Text with approximated math
        """
        # Look for common math patterns and add [MATH] flag
        math_patterns = [
            r'∑', r'∫', r'∂', r'√', r'∞', r'π', r'θ',  # Math symbols
            r'α', r'β', r'γ', r'δ',  # Greek letters
            r'\$_\w+\$_',  # LaTeX-style math: $_variable_$
            r'\\\w+',  # LaTeX commands
        ]

        has_math = any(re.search(pattern, text) for pattern in math_patterns)

        if has_math:
            # Add [MATH] flag at the end
            return text + " [MATH]"

        return text

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison (case insensitive, remove extra spaces).

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        return re.sub(r'\s+', ' ', text.lower().strip())

    def _final_cleanup(self, text: str) -> str:
        """
        Apply final cleanup to the entire document.

        Args:
            text: Complete tagged document

        Returns:
            Final cleaned document
        """
        # Remove consecutive empty lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove empty blocks
        text = re.sub(r'\w+:\s*\n', '', text)  # Remove empty type-tagged lines

        # Ensure proper spacing between blocks
        text = re.sub(r'(\S):\s*(\w+:)', r'\1\n\n\2', text)

        return text.strip()

    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def extract_and_clean_pdf(pdf_path: str) -> str:
    """
    Extract and clean PDF content with structure tagging.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Clean tagged text string

    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    print(f"Extracting content from: {pdf_path}")

    with PDFExtractor(pdf_path) as extractor:
        clean_text = extractor.extract_and_clean()

    print(f"Extraction complete: {len(clean_text)} characters")

    return clean_text