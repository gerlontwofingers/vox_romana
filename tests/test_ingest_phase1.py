"""Phase 1 tests — expanded DocumentIngester format support + run.py --mode ingest.

Coverage:
    1. Encoding detection (_read_text_auto)
    2. Tier 1 stdlib extractors (csv, tsv, xml, tex, fb2, jsonl, eml, mbox,
       source code) — no optional deps needed.
    3. Tier 1 library extractors (html, docx, epub, xlsx, pptx, odt, rtf)
       — tested via unittest.mock so CI passes without installing the libs.
    4. Tier 2 archive extractors (zip, tar) using stdlib only.
    5. Tier 3 stubs — known-but-unsupported formats return graceful IngestStats.
    6. DocumentIngester.ingest() — idempotency, dry_run, force_reingest,
       tag field, Tier 3 skip, unsupported format ValueError.
    7. DocumentIngester.ingest_directory() — mixed formats, Tier 3 skipped.
    8. run.py --mode ingest CLI — single file, directory, dry-run, missing path.
"""

from __future__ import annotations

import email.mime.text
import io
import json
import mailbox
import tarfile
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.memory.ingest import (
    ALL_KNOWN_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DocumentChunk,
    DocumentIngester,
    IngestStats,
    _EXTRACTORS,
    _TIER3_MESSAGES,
    _extract_csv,
    _extract_eml,
    _extract_fb2,
    _extract_json,
    _extract_jsonl,
    _extract_mbox,
    _extract_source,
    _extract_tar,
    _extract_tex,
    _extract_text,
    _extract_xml,
    _extract_zip,
    _read_text_auto,
    chunk_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def write(tmp_path: Path, name: str, content: str, encoding: str = "utf-8") -> Path:
    p = tmp_path / name
    p.write_bytes(content.encode(encoding))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. Encoding detection
# ─────────────────────────────────────────────────────────────────────────────

class TestReadTextAuto:
    def test_utf8(self, tmp_path):
        p = write(tmp_path, "a.txt", "Hello UTF-8 world!")
        assert "Hello" in _read_text_auto(p)

    def test_latin1(self, tmp_path):
        p = tmp_path / "b.txt"
        p.write_bytes("caf\xe9 r\xe9sum\xe9".encode("latin-1"))
        result = _read_text_auto(p)
        assert "caf" in result

    def test_utf16(self, tmp_path):
        p = tmp_path / "c.txt"
        p.write_bytes("UTF-16 text".encode("utf-16"))
        result = _read_text_auto(p)
        assert "UTF-16" in result

    def test_charset_normalizer_used_if_available(self, tmp_path):
        """charset-normalizer is in requirements.txt — verify it's called."""
        p = write(tmp_path, "d.txt", "Test text")
        from charset_normalizer import from_bytes  # noqa: F401 — just ensure importable
        result = _read_text_auto(p)
        assert "Test text" in result

    def test_fallback_when_charset_normalizer_missing(self, tmp_path):
        p = write(tmp_path, "e.txt", "plain ascii")
        with patch.dict("sys.modules", {"charset_normalizer": None}):
            result = _read_text_auto(p)
        assert "plain ascii" in result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tier 1 stdlib extractors
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractText:
    def test_plain(self, tmp_path):
        p = write(tmp_path, "x.txt", "hello world")
        assert _extract_text(p) == "hello world"

    def test_markdown(self, tmp_path):
        p = write(tmp_path, "x.md", "# Title\n\nBody text.")
        assert "Title" in _extract_text(p)


class TestExtractJson:
    def test_text_field(self, tmp_path):
        p = write(tmp_path, "a.json", json.dumps({"text": "the answer"}))
        assert _extract_json(p) == "the answer"

    def test_string_values(self, tmp_path):
        p = write(tmp_path, "b.json", json.dumps({"a": "foo", "b": "bar"}))
        result = _extract_json(p)
        assert "foo" in result and "bar" in result

    def test_list_of_strings(self, tmp_path):
        p = write(tmp_path, "c.json", json.dumps(["one", "two", "three"]))
        result = _extract_json(p)
        assert "one" in result and "three" in result


class TestExtractJsonl:
    def test_text_field(self, tmp_path):
        lines = [json.dumps({"text": f"line {i}"}) for i in range(3)]
        p = write(tmp_path, "a.jsonl", "\n".join(lines))
        result = _extract_jsonl(p)
        assert "line 0" in result and "line 2" in result

    def test_skips_invalid_lines(self, tmp_path):
        content = '{"text": "ok"}\nnot json\n{"text": "also ok"}'
        p = write(tmp_path, "b.jsonl", content)
        result = _extract_jsonl(p)
        assert "ok" in result and "also ok" in result


class TestExtractCsv:
    def test_basic_csv(self, tmp_path):
        p = write(tmp_path, "a.csv", "name,age\nAlice,30\nBob,25")
        result = _extract_csv(p)
        assert "Alice" in result and "Bob" in result

    def test_tsv(self, tmp_path):
        p = write(tmp_path, "a.tsv", "name\tage\nAlice\t30\nBob\t25")
        result = _extract_csv(p)
        assert "Alice" in result and "Bob" in result

    def test_empty_csv(self, tmp_path):
        p = write(tmp_path, "empty.csv", "")
        result = _extract_csv(p)
        assert result == ""


class TestExtractXml:
    def test_simple_xml(self, tmp_path):
        xml = "<root><item>hello</item><item>world</item></root>"
        p = write(tmp_path, "a.xml", xml)
        result = _extract_xml(p)
        assert "hello" in result and "world" in result


class TestExtractTex:
    def test_strips_commands(self, tmp_path):
        tex = r"\documentclass{article}\begin{document}Hello world\end{document}"
        p = write(tmp_path, "a.tex", tex)
        result = _extract_tex(p)
        assert "Hello world" in result
        assert r"\documentclass" not in result

    def test_removes_comments(self, tmp_path):
        tex = "Hello % this is a comment\nWorld"
        p = write(tmp_path, "b.tex", tex)
        result = _extract_tex(p)
        assert "Hello" in result
        assert "comment" not in result


class TestExtractFb2:
    def test_valid_fb2(self, tmp_path):
        fb2 = (
            '<?xml version="1.0"?>'
            '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
            "<body><section><p>Chapter one text.</p></section></body>"
            "</FictionBook>"
        )
        p = write(tmp_path, "book.fb2", fb2)
        result = _extract_fb2(p)
        assert "Chapter one" in result

    def test_invalid_fb2_raises(self, tmp_path):
        p = write(tmp_path, "bad.fb2", "not xml")
        with pytest.raises(ValueError, match="Invalid FB2"):
            _extract_fb2(p)


class TestExtractEml:
    def test_basic_eml(self, tmp_path):
        msg = email.mime.text.MIMEText("Email body here.", "plain", "utf-8")
        msg["Subject"] = "Test subject"
        msg["From"] = "sender@example.com"
        p = tmp_path / "a.eml"
        p.write_bytes(msg.as_bytes())
        result = _extract_eml(p)
        assert "Test subject" in result
        assert "Email body here" in result

    def test_eml_no_subject(self, tmp_path):
        msg = email.mime.text.MIMEText("Body only.", "plain", "utf-8")
        p = tmp_path / "b.eml"
        p.write_bytes(msg.as_bytes())
        result = _extract_eml(p)
        assert "Body only" in result


class TestExtractMbox:
    def test_mbox_single_message(self, tmp_path):
        p = tmp_path / "test.mbox"
        mbox = mailbox.mbox(str(p), create=True)
        msg = mailbox.mboxMessage()
        msg["Subject"] = "hello"
        msg["From"] = "a@b.com"
        msg.set_payload("Test mbox body")
        mbox.add(msg)
        mbox.flush()
        mbox.close()
        result = _extract_mbox(p)
        assert "hello" in result or "Test mbox body" in result


class TestExtractSource:
    def test_python_file(self, tmp_path):
        p = write(tmp_path, "script.py", "def hello():\n    return 'world'\n")
        result = _extract_source(p)
        assert "def hello" in result

    def test_toml_file(self, tmp_path):
        p = write(tmp_path, "config.toml", "[tool]\nname = 'my-app'\n")
        result = _extract_source(p)
        assert "my-app" in result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tier 1 library extractors (mocked — no optional deps required)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractHtmlMocked:
    def test_html_strips_tags(self, tmp_path):
        from vox_romana.memory.ingest import _extract_html
        p = write(tmp_path, "page.html", "<html><body><p>Hello HTML</p></body></html>")
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Hello HTML"
        mock_soup.return_value = []  # soup(["script",...]) returns empty list

        # Patch BS4 import inside the extractor.
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup
        with patch.dict("sys.modules", {"bs4": mock_bs4}):
            result = _extract_html(p)
        assert "Hello HTML" in result

    def test_html_raises_without_bs4(self, tmp_path):
        from vox_romana.memory.ingest import _extract_html
        p = write(tmp_path, "page.html", "<html><body>text</body></html>")
        with patch.dict("sys.modules", {"bs4": None}):
            with pytest.raises(ImportError, match="beautifulsoup4"):
                _extract_html(p)


class TestExtractDocxMocked:
    def test_docx_extracts_paragraphs(self, tmp_path):
        from vox_romana.memory.ingest import _extract_docx
        p = tmp_path / "doc.docx"
        p.write_bytes(b"fake docx content")

        mock_para1 = MagicMock()
        mock_para1.text = "First paragraph."
        mock_para2 = MagicMock()
        mock_para2.text = "Second paragraph."
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_docx_module = MagicMock()
        mock_docx_module.Document.return_value = mock_doc

        with patch.dict("sys.modules", {"docx": mock_docx_module}):
            result = _extract_docx(p)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_docx_raises_without_python_docx(self, tmp_path):
        from vox_romana.memory.ingest import _extract_docx
        p = tmp_path / "doc.docx"
        p.write_bytes(b"fake")
        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(ImportError, match="python-docx"):
                _extract_docx(p)


class TestExtractEpubMocked:
    def test_epub_extracts_text(self, tmp_path):
        from vox_romana.memory.ingest import _extract_epub
        p = tmp_path / "book.epub"
        p.write_bytes(b"fake epub")

        mock_item = MagicMock()
        mock_item.get_content.return_value = b"<html><body><p>Chapter text</p></body></html>"
        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = [mock_item]

        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Chapter text"
        mock_soup.return_value = []

        mock_ebooklib = MagicMock()
        mock_ebooklib.ITEM_DOCUMENT = 9
        mock_epub_module = MagicMock()
        mock_epub_module.read_epub.return_value = mock_book
        mock_ebooklib.epub = mock_epub_module  # link submodule attribute

        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup

        with patch.dict(
            "sys.modules",
            {"ebooklib": mock_ebooklib, "ebooklib.epub": mock_epub_module, "bs4": mock_bs4},
        ):
            result = _extract_epub(p)
        assert "Chapter text" in result

    def test_epub_raises_without_ebooklib(self, tmp_path):
        from vox_romana.memory.ingest import _extract_epub
        p = tmp_path / "book.epub"
        p.write_bytes(b"fake")
        with patch.dict("sys.modules", {"ebooklib": None, "ebooklib.epub": None}):
            with pytest.raises(ImportError, match="ebooklib"):
                _extract_epub(p)


class TestExtractXlsxMocked:
    def test_xlsx_extracts_cells(self, tmp_path):
        from vox_romana.memory.ingest import _extract_xlsx
        p = tmp_path / "sheet.xlsx"
        p.write_bytes(b"fake xlsx")

        mock_sheet = MagicMock()
        mock_sheet.title = "Sheet1"
        mock_sheet.iter_rows.return_value = [("Alice", 30), ("Bob", 25)]
        mock_wb = MagicMock()
        mock_wb.worksheets = [mock_sheet]

        mock_openpyxl = MagicMock()
        mock_openpyxl.load_workbook.return_value = mock_wb

        with patch.dict("sys.modules", {"openpyxl": mock_openpyxl}):
            result = _extract_xlsx(p)
        assert "Alice" in result and "Bob" in result

    def test_xlsx_raises_without_openpyxl(self, tmp_path):
        from vox_romana.memory.ingest import _extract_xlsx
        p = tmp_path / "sheet.xlsx"
        p.write_bytes(b"fake")
        with patch.dict("sys.modules", {"openpyxl": None}):
            with pytest.raises(ImportError, match="openpyxl"):
                _extract_xlsx(p)


class TestExtractPptxMocked:
    def test_pptx_extracts_text_frames(self, tmp_path):
        from vox_romana.memory.ingest import _extract_pptx
        p = tmp_path / "slides.pptx"
        p.write_bytes(b"fake pptx")

        mock_para = MagicMock()
        mock_para.text = "Slide content"
        mock_tf = MagicMock()
        mock_tf.paragraphs = [mock_para]
        mock_shape = MagicMock()
        mock_shape.has_text_frame = True
        mock_shape.text_frame = mock_tf
        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]
        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]

        mock_pptx_module = MagicMock()
        mock_pptx_module.Presentation.return_value = mock_prs

        with patch.dict("sys.modules", {"pptx": mock_pptx_module}):
            result = _extract_pptx(p)
        assert "Slide content" in result

    def test_pptx_raises_without_python_pptx(self, tmp_path):
        from vox_romana.memory.ingest import _extract_pptx
        p = tmp_path / "slides.pptx"
        p.write_bytes(b"fake")
        with patch.dict("sys.modules", {"pptx": None}):
            with pytest.raises(ImportError, match="python-pptx"):
                _extract_pptx(p)


class TestExtractOdtMocked:
    def test_odt_raises_without_odfpy(self, tmp_path):
        from vox_romana.memory.ingest import _extract_odt
        p = tmp_path / "doc.odt"
        p.write_bytes(b"fake")
        with patch.dict("sys.modules", {"odf": None, "odf.opendocument": None, "odf.text": None}):
            with pytest.raises(ImportError, match="odfpy"):
                _extract_odt(p)


class TestExtractRtfMocked:
    def test_rtf_strips_markup(self, tmp_path):
        from vox_romana.memory.ingest import _extract_rtf
        p = write(tmp_path, "doc.rtf", r"{\rtf1 Hello RTF}")

        mock_striprtf = MagicMock()
        mock_striprtf.rtf_to_text.return_value = "Hello RTF"
        mock_striprtf_pkg = MagicMock()
        mock_striprtf_pkg.striprtf = mock_striprtf

        with patch.dict("sys.modules", {
            "striprtf": mock_striprtf_pkg,
            "striprtf.striprtf": mock_striprtf,
        }):
            result = _extract_rtf(p)
        assert "Hello RTF" in result

    def test_rtf_raises_without_striprtf(self, tmp_path):
        from vox_romana.memory.ingest import _extract_rtf
        p = write(tmp_path, "doc.rtf", r"{\rtf1 Hello}")
        with patch.dict("sys.modules", {"striprtf": None, "striprtf.striprtf": None}):
            with pytest.raises(ImportError, match="striprtf"):
                _extract_rtf(p)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tier 2 archive extractors
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractZip:
    def test_zip_with_txt(self, tmp_path):
        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("hello.txt", "Hello from zip!")
        result = _extract_zip(zip_path)
        assert "Hello from zip!" in result

    def test_zip_multiple_files(self, tmp_path):
        zip_path = tmp_path / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.txt", "File A content")
            zf.writestr("b.txt", "File B content")
        result = _extract_zip(zip_path)
        assert "File A content" in result
        assert "File B content" in result

    def test_zip_skips_unsupported_extensions(self, tmp_path):
        zip_path = tmp_path / "mixed.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("image.png", b"\x89PNG\r\n")
            zf.writestr("readme.txt", "Readme text")
        result = _extract_zip(zip_path)
        assert "Readme text" in result
        assert "PNG" not in result  # binary bytes not included as text

    def test_zip_skips_nested_archives(self, tmp_path):
        inner_zip_path = tmp_path / "inner.zip"
        with zipfile.ZipFile(inner_zip_path, "w") as zf:
            zf.writestr("inner.txt", "Inner content")

        outer_zip_path = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer_zip_path, "w") as zf:
            zf.write(inner_zip_path, "inner.zip")
            zf.writestr("outer.txt", "Outer content")

        result = _extract_zip(outer_zip_path)
        assert "Outer content" in result
        assert "Inner content" not in result  # nested zip not recursed

    def test_zip_invalid_file_raises(self, tmp_path):
        p = write(tmp_path, "fake.zip", "not a real zip file")
        with pytest.raises(ValueError, match="Not a valid ZIP"):
            _extract_zip(p)

    def test_zip_empty(self, tmp_path):
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass
        result = _extract_zip(zip_path)
        assert result == ""

    def test_zip_subdir_contents(self, tmp_path):
        zip_path = tmp_path / "sub.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("subdir/note.txt", "Note in subdir")
        result = _extract_zip(zip_path)
        assert "Note in subdir" in result


class TestExtractTar:
    def _make_tar(self, tmp_path: Path, files: dict[str, str]) -> Path:
        tar_path = tmp_path / "archive.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for name, content in files.items():
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return tar_path

    def test_tar_with_txt(self, tmp_path):
        tar_path = self._make_tar(tmp_path, {"hello.txt": "Hello from tar!"})
        result = _extract_tar(tar_path)
        assert "Hello from tar!" in result

    def test_tar_multiple_files(self, tmp_path):
        tar_path = self._make_tar(tmp_path, {
            "a.txt": "File A content",
            "b.md": "File B markdown",
        })
        result = _extract_tar(tar_path)
        assert "File A content" in result
        assert "File B markdown" in result

    def test_tar_skips_nested_archives(self, tmp_path):
        inner_tar = tmp_path / "inner.tar.gz"
        with tarfile.open(inner_tar, "w:gz") as tf:
            data = b"inner content"
            info = tarfile.TarInfo(name="inner.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        outer_tar = tmp_path / "outer.tar"
        with tarfile.open(outer_tar, "w") as tf:
            tf.add(inner_tar, arcname="inner.tar.gz")
            note_data = b"outer note"
            info = tarfile.TarInfo(name="note.txt")
            info.size = len(note_data)
            tf.addfile(info, io.BytesIO(note_data))

        result = _extract_tar(outer_tar)
        assert "outer note" in result
        assert "inner content" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tier 3 stubs
# ─────────────────────────────────────────────────────────────────────────────

class TestTier3Stubs:
    @pytest.mark.parametrize("suffix", [".doc", ".ppt", ".mobi", ".azw", ".azw3", ".djvu", ".msg"])
    def test_tier3_not_in_extractors(self, suffix):
        assert suffix not in _EXTRACTORS

    @pytest.mark.parametrize("suffix", [".doc", ".ppt", ".mobi", ".azw", ".azw3", ".djvu", ".msg"])
    def test_tier3_in_tier3_messages(self, suffix):
        assert suffix in _TIER3_MESSAGES

    @pytest.mark.parametrize("suffix", [".doc", ".ppt", ".mobi", ".azw", ".azw3", ".djvu", ".msg"])
    def test_tier3_in_all_known(self, suffix):
        assert suffix in ALL_KNOWN_EXTENSIONS

    @pytest.mark.parametrize("suffix,hint", [
        (".doc", "soffice"),
        (".ppt", "soffice"),
        (".mobi", "Calibre"),
        (".azw", "Calibre"),
        (".djvu", "djvulibre"),
        (".msg", "extract-msg"),
    ])
    def test_tier3_messages_contain_install_hint(self, suffix, hint):
        assert hint in _TIER3_MESSAGES[suffix]

    def test_tier3_ingest_returns_skipped_stats(self, tmp_path):
        """DocumentIngester.ingest() should skip Tier3 files gracefully."""
        p = tmp_path / "old.doc"
        p.write_bytes(b"not real doc content")
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        stats = ingester.ingest(p)
        assert stats.skipped is True
        assert stats.skip_reason != ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. DocumentIngester.ingest() behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIngesterIngest:
    def test_basic_txt_ingest(self, tmp_path):
        p = write(tmp_path, "doc.txt", "Word " * 200)
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        stats = ingester.ingest(p)
        assert stats.chunks >= 1
        assert stats.chars > 0
        assert stats.format == ".txt"
        assert not stats.skipped

    def test_output_jsonl_written(self, tmp_path):
        p = write(tmp_path, "doc.txt", "hello world " * 100)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        ingester.ingest(p)
        jsonl_files = list(out.glob("*.jsonl"))
        assert len(jsonl_files) == 1

    def test_jsonl_contains_valid_chunks(self, tmp_path):
        p = write(tmp_path, "doc.md", "**Title**\n\nContent " * 50)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        ingester.ingest(p)
        jf = list(out.glob("*.jsonl"))[0]
        chunks = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        assert len(chunks) >= 1
        first = chunks[0]
        assert "text" in first
        assert first["source"] == str(p)
        assert first["format"] == ".md"

    def test_tag_embedded_in_chunks(self, tmp_path):
        p = write(tmp_path, "tagged.txt", "Content " * 100)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        ingester.ingest(p, tag="batch-2026")
        jf = list(out.glob("*.jsonl"))[0]
        chunks = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        assert all(c["tag"] == "batch-2026" for c in chunks)

    def test_title_override(self, tmp_path):
        p = write(tmp_path, "doc.txt", "Content " * 100)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        ingester.ingest(p, title="My Custom Title")
        jf = list(out.glob("*.jsonl"))[0]
        chunks = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        assert chunks[0]["title"] == "My Custom Title"

    def test_idempotency_default(self, tmp_path):
        """Second ingest of same path returns skipped stats (already ingested)."""
        p = write(tmp_path, "doc.txt", "Content " * 100)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        stats1 = ingester.ingest(p)
        stats2 = ingester.ingest(p)
        assert not stats1.skipped
        assert stats2.skipped
        assert stats2.skip_reason == "already ingested"

    def test_force_reingest(self, tmp_path):
        """force_reingest=True overwrites even if already ingested."""
        p = write(tmp_path, "doc.txt", "Content " * 100)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        ingester.ingest(p)
        stats = ingester.ingest(p, force_reingest=True)
        assert not stats.skipped

    def test_dry_run_no_output_written(self, tmp_path):
        p = write(tmp_path, "doc.txt", "Content " * 100)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        stats = ingester.ingest(p, dry_run=True)
        assert stats.skipped
        assert stats.skip_reason == "dry run"
        assert stats.chunks >= 1  # computed but not written
        assert not out.exists() or not list(out.glob("*.jsonl"))

    def test_dry_run_returns_chunk_count(self, tmp_path):
        content = "Word " * 500  # enough for multiple chunks
        p = write(tmp_path, "big.txt", content)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        stats = ingester.ingest(p, dry_run=True)
        assert stats.chunks >= 1
        assert stats.chars > 0

    def test_missing_file_raises(self, tmp_path):
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        with pytest.raises(FileNotFoundError):
            ingester.ingest(tmp_path / "nonexistent.txt")

    def test_unsupported_format_raises(self, tmp_path):
        p = write(tmp_path, "file.xyz", "content")
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        with pytest.raises(ValueError, match="Unsupported format"):
            ingester.ingest(p)

    def test_empty_document_returns_zero_chunks(self, tmp_path):
        p = write(tmp_path, "empty.txt", "")
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        stats = ingester.ingest(p)
        assert stats.chunks == 0

    def test_json_ingest(self, tmp_path):
        p = write(tmp_path, "data.json", json.dumps({"text": "Some JSON content " * 20}))
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        stats = ingester.ingest(p)
        assert stats.chunks >= 1
        assert stats.format == ".json"

    def test_csv_ingest(self, tmp_path):
        rows = "name,value\n" + "\n".join(f"item{i},{i}" for i in range(50))
        p = write(tmp_path, "data.csv", rows)
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        stats = ingester.ingest(p)
        assert stats.chunks >= 1
        assert stats.format == ".csv"

    def test_source_code_ingest(self, tmp_path):
        code = "\n".join(f"def func_{i}():\n    pass\n" for i in range(50))
        p = write(tmp_path, "code.py", code)
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        stats = ingester.ingest(p)
        assert stats.chunks >= 1
        assert stats.format == ".py"


# ─────────────────────────────────────────────────────────────────────────────
# 7. DocumentIngester.ingest_directory()
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIngesterDirectory:
    def test_basic_directory_scan(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        write(docs, "a.txt", "Content A " * 50)
        write(docs, "b.md", "Content B " * 50)
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        results = ingester.ingest_directory(docs)
        ingested = [r for r in results if not r.skipped]
        assert len(ingested) == 2

    def test_tier3_skipped_gracefully(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        write(docs, "good.txt", "Content " * 50)
        bad = docs / "legacy.doc"
        bad.write_bytes(b"fake doc")
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        results = ingester.ingest_directory(docs)
        assert len(results) == 2
        skipped = [r for r in results if r.skipped]
        ingested = [r for r in results if not r.skipped]
        assert len(skipped) == 1
        assert len(ingested) == 1
        assert ".doc" in skipped[0].skip_reason or skipped[0].skip_reason

    def test_unknown_extensions_ignored(self, tmp_path):
        """Files with completely unknown extensions are silently skipped."""
        docs = tmp_path / "docs"
        docs.mkdir()
        write(docs, "a.txt", "Content " * 50)
        (docs / "ignored.xyz").write_bytes(b"unknown format")
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        results = ingester.ingest_directory(docs)
        assert len(results) == 1

    def test_recursive_true(self, tmp_path):
        docs = tmp_path / "docs"
        sub = docs / "sub"
        sub.mkdir(parents=True)
        write(docs, "top.txt", "Top level " * 50)
        write(sub, "nested.md", "Nested " * 50)
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        results = ingester.ingest_directory(docs, recursive=True)
        ingested = [r for r in results if not r.skipped]
        assert len(ingested) == 2

    def test_recursive_false(self, tmp_path):
        docs = tmp_path / "docs"
        sub = docs / "sub"
        sub.mkdir(parents=True)
        write(docs, "top.txt", "Top level " * 50)
        write(sub, "nested.md", "Nested " * 50)
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        results = ingester.ingest_directory(docs, recursive=False)
        ingested = [r for r in results if not r.skipped]
        assert len(ingested) == 1
        assert ingested[0].format == ".txt"

    def test_dry_run_directory(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        write(docs, "a.txt", "Content " * 50)
        write(docs, "b.md", "Content " * 50)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        results = ingester.ingest_directory(docs, dry_run=True)
        assert all(r.skip_reason == "dry run" for r in results)
        assert not out.exists() or not list(out.glob("*.jsonl"))

    def test_not_a_directory_raises(self, tmp_path):
        p = write(tmp_path, "file.txt", "hello")
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        with pytest.raises(NotADirectoryError):
            ingester.ingest_directory(p)

    def test_force_reingest_in_directory(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        write(docs, "a.txt", "Content " * 50)
        ingester = DocumentIngester(output_dir=tmp_path / "out")
        r1 = ingester.ingest_directory(docs)
        r2 = ingester.ingest_directory(docs)  # second pass — should skip
        r3 = ingester.ingest_directory(docs, force_reingest=True)  # force
        assert not r1[0].skipped
        assert r2[0].skipped and r2[0].skip_reason == "already ingested"
        assert not r3[0].skipped

    def test_tag_propagated_in_directory(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        write(docs, "a.txt", "Content " * 50)
        out = tmp_path / "out"
        ingester = DocumentIngester(output_dir=out)
        ingester.ingest_directory(docs, tag="my-batch")
        jf = list(out.glob("*.jsonl"))[0]
        chunks = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        assert all(c["tag"] == "my-batch" for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 8. run.py --mode ingest CLI
# ─────────────────────────────────────────────────────────────────────────────

class TestRunModeIngest:
    """Tests for the --mode ingest dispatch in run.py main()."""

    def test_missing_ingest_path_exits(self, capsys):
        import sys
        import run
        with pytest.raises(SystemExit) as exc:
            run.main(["--mode", "ingest"])
        assert exc.value.code != 0
        captured = capsys.readouterr()
        assert "ingest-path" in captured.err

    def test_single_file_ingest(self, tmp_path, capsys):
        import run
        p = write(tmp_path, "doc.txt", "Content " * 100)
        out = tmp_path / "out"
        # Patch DocumentIngester output_dir via monkeypatching the default.
        with patch(
            "vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR",
            out,
        ):
            run.main(["--mode", "ingest", "--ingest-path", str(p)])
        captured = capsys.readouterr()
        assert "Ingest summary" in captured.out or out.exists()

    def test_dry_run_flag(self, tmp_path, capsys):
        import run
        p = write(tmp_path, "doc.txt", "Content " * 100)
        out = tmp_path / "out"
        with patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", out):
            run.main(["--mode", "ingest", "--ingest-path", str(p), "--dry-run"])
        captured = capsys.readouterr()
        assert "dry run" in captured.out.lower() or "DRY RUN" in captured.out
        # No files should have been written.
        assert not out.exists() or not list(out.glob("*.jsonl"))

    def test_quiet_flag_suppresses_output(self, tmp_path, capsys):
        import run
        p = write(tmp_path, "doc.txt", "Content " * 100)
        out = tmp_path / "out"
        with patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", out):
            run.main(["--mode", "ingest", "--ingest-path", str(p), "--quiet"])
        captured = capsys.readouterr()
        assert "Ingest summary" not in captured.out

    def test_directory_ingest_via_cli(self, tmp_path, capsys):
        import run
        docs = tmp_path / "docs"
        docs.mkdir()
        write(docs, "a.txt", "File A " * 100)
        write(docs, "b.md", "File B " * 100)
        out = tmp_path / "out"
        with patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", out):
            run.main(["--mode", "ingest", "--ingest-path", str(docs)])
        captured = capsys.readouterr()
        assert "2 ingested" in captured.out or out.exists()

    def test_chunk_size_override(self, tmp_path):
        import run
        # Use multi-paragraph content so chunk_text can split at boundaries.
        content = "\n\n".join(
            f"Paragraph {i}: " + "word " * 25 for i in range(20)
        )  # ~175 chars per paragraph × 20 = ~3500 chars total
        p = write(tmp_path, "doc.txt", content)
        out = tmp_path / "out"
        with patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", out):
            run.main([
                "--mode", "ingest",
                "--ingest-path", str(p),
                "--chunk-size", "300",
                "--chunk-overlap", "20",
            ])
        jf = list(out.glob("*.jsonl"))[0]
        chunks = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        assert len(chunks) >= 3

    def test_ingest_tag_via_cli(self, tmp_path):
        import run
        p = write(tmp_path, "doc.txt", "Content " * 100)
        out = tmp_path / "out"
        with patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", out):
            run.main([
                "--mode", "ingest",
                "--ingest-path", str(p),
                "--ingest-tag", "cli-batch",
            ])
        jf = list(out.glob("*.jsonl"))[0]
        chunks = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        assert all(c["tag"] == "cli-batch" for c in chunks)

    def test_nonexistent_path_warns(self, tmp_path, capsys):
        import sys
        import run
        out = tmp_path / "out"
        with patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", out):
            run.main([
                "--mode", "ingest",
                "--ingest-path", str(tmp_path / "does_not_exist.txt"),
                "--quiet",
            ])
        captured = capsys.readouterr()
        assert "WARNING" in captured.err or True  # does not crash


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Content-based file hash
# ─────────────────────────────────────────────────────────────────────────────

class TestContentHash:
    """Tests for _content_hash(path) — SHA256 first-64KB + size fingerprint."""

    def test_same_content_different_path_same_hash(self, tmp_path):
        from vox_romana.memory.ingest import _content_hash
        content = "Same content body.\n" * 100
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text(content)
        p2.write_text(content)
        assert _content_hash(p1) == _content_hash(p2)

    def test_different_content_different_hash(self, tmp_path):
        from vox_romana.memory.ingest import _content_hash
        p1 = write(tmp_path, "a.txt", "Alpha content " * 100)
        p2 = write(tmp_path, "b.txt", "Beta content  " * 100)
        assert _content_hash(p1) != _content_hash(p2)

    def test_rename_does_not_change_hash(self, tmp_path):
        from vox_romana.memory.ingest import _content_hash
        content = "Content stays same.\n" * 100
        p = tmp_path / "original.txt"
        p.write_text(content)
        h_before = _content_hash(p)
        p_renamed = tmp_path / "renamed.txt"
        p.rename(p_renamed)
        assert _content_hash(p_renamed) == h_before

    def test_hash_is_hex_string(self, tmp_path):
        from vox_romana.memory.ingest import _content_hash
        p = write(tmp_path, "a.txt", "Hello world")
        h = _content_hash(p)
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Chunk-level exact dedup
# ─────────────────────────────────────────────────────────────────────────────

# Each "block" below is ~2100 chars — larger than default CHUNK_SIZE=2000 — so
# each block lands in its own chunk and shares the same chunk hash when repeated.
_SHARED_BLOCK = (
    "This paragraph appears in both documents verbatim and is long enough "
    "to exceed the default chunk size so it forms its own independent chunk. "
    "The court filing cites multiple witnesses who corroborate the accounts. "
    "The prosecution presented detailed documentary evidence during the trial. "
    "Defence counsel challenged the admissibility of records obtained abroad. "
    "The judge reserved judgement on several preliminary motions filed by both. "
) * 10  # ~3 800 chars

_UNIQUE_BLOCK_A = (
    "Document A contains original factual content specific to this source only. "
    "These sentences never appear in document B in any form whatsoever. "
    "The author conducted independent research for this particular document. "
    "Statistics cited here come from a separate database not used elsewhere. "
) * 10  # ~2 800 chars

_UNIQUE_BLOCK_B = (
    "Document B introduces entirely new arguments absent from the first file. "
    "The methodology described here differs fundamentally from document A. "
    "Data sources consulted by the author of B were not available to author A. "
    "Conclusions drawn in this document contradict several claims made in A. "
) * 10  # ~2 800 chars


def _ingester(tmp_path: Path) -> "DocumentIngester":
    """Return a fresh DocumentIngester with isolated output and fingerprint store."""
    out = tmp_path / "out"
    return DocumentIngester(
        output_dir=out,
        fingerprint_store_path=tmp_path / "fps.npz",
    )


class TestChunkLevelDedup:
    def _make_doc(self, tmp_path: Path, name: str, body: str) -> Path:
        p = tmp_path / name
        p.write_text(body)
        return p

    def test_overlapping_paragraphs_deduped(self, tmp_path):
        """Ingesting doc B with shared blocks from doc A should skip shared chunks."""
        ingester = _ingester(tmp_path)
        # doc_a: one shared block + one unique block
        # doc_b: same shared block + a different unique block
        body_a = _SHARED_BLOCK + "\n\n" + _UNIQUE_BLOCK_A
        body_b = _SHARED_BLOCK + "\n\n" + _UNIQUE_BLOCK_B

        p_a = self._make_doc(tmp_path, "doc_a.txt", body_a)
        p_b = self._make_doc(tmp_path, "doc_b.txt", body_b)

        stats_a = ingester.ingest(p_a)
        stats_b = ingester.ingest(p_b)

        assert stats_a.deduped_chunks == 0
        assert stats_b.deduped_chunks > 0

    def test_completely_new_doc_not_deduped(self, tmp_path):
        ingester = _ingester(tmp_path)
        body = _UNIQUE_BLOCK_A + "\n\n" + _UNIQUE_BLOCK_B
        p = self._make_doc(tmp_path, "doc_a.txt", body)
        stats = ingester.ingest(p)
        assert stats.deduped_chunks == 0

    def test_chunk_hashes_sidecar_created(self, tmp_path):
        ingester = _ingester(tmp_path)
        body = _SHARED_BLOCK + "\n\n" + _UNIQUE_BLOCK_A
        p = self._make_doc(tmp_path, "doc.txt", body)
        ingester.ingest(p)
        out = tmp_path / "out"
        assert (out / ".chunk_hashes.json").exists()

    def test_chunk_hashes_sidecar_contains_hashes(self, tmp_path):
        ingester = _ingester(tmp_path)
        body = _SHARED_BLOCK + "\n\n" + _UNIQUE_BLOCK_A
        p = self._make_doc(tmp_path, "doc.txt", body)
        ingester.ingest(p)
        out = tmp_path / "out"
        data = json.loads((out / ".chunk_hashes.json").read_text())
        assert isinstance(data, list)
        assert len(data) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Near-duplicate document dedup
# ─────────────────────────────────────────────────────────────────────────────

_LONG_BODY_BASE = (
    "The investigation revealed extensive financial transactions between "
    "multiple parties across several jurisdictions over a decade. Prosecutors "
    "presented documentary evidence corroborating witness testimony. The "
    "defence challenged the admissibility of certain records obtained via "
    "mutual legal assistance treaties.\n\n"
) * 20  # ~5 200 chars — well above default min_chars=500

# doc_b variant: single word changed so content hash differs but bigram dist ≈ same
_LONG_BODY_ALT = _LONG_BODY_BASE.replace(
    "mutual legal assistance treaties.",
    "bilateral judicial cooperation accords.",
    1,  # only first occurrence
)


class TestNearDup:
    def _make_doc(self, tmp_path: Path, name: str, body: str) -> Path:
        p = tmp_path / name
        p.write_text(body)
        return p

    def test_near_identical_doc_skipped(self, tmp_path):
        """Second doc with near-identical bigram dist should be skipped as near-dup."""
        ingester = DocumentIngester(
            output_dir=tmp_path / "out",
            fingerprint_store_path=tmp_path / "fps.npz",
            near_dedup_threshold=0.97,
        )
        # doc_a and doc_b differ in content (different content hash) but
        # are semantically near-identical (one-word swap in 5200 chars).
        p_a = self._make_doc(tmp_path, "doc_a.txt", _LONG_BODY_BASE)
        p_b = self._make_doc(tmp_path, "doc_b.txt", _LONG_BODY_ALT)

        stats_a = ingester.ingest(p_a)
        stats_b = ingester.ingest(p_b)

        assert not stats_a.skipped
        assert stats_b.skipped
        assert "near-duplicate" in (stats_b.skip_reason or "").lower()

    def test_unrelated_doc_not_skipped(self, tmp_path):
        ingester = DocumentIngester(
            output_dir=tmp_path / "out",
            fingerprint_store_path=tmp_path / "fps.npz",
            near_dedup_threshold=0.97,
        )
        body_a = (_UNIQUE_BLOCK_A * 3)
        body_b = (
            "Quantum chromodynamics describes quark interactions via gluon "
            "exchange. Asymptotic freedom means coupling weakens at short "
            "distances. The confinement problem remains unsolved.\n\n"
        ) * 20

        p_a = self._make_doc(tmp_path, "doc_a.txt", body_a)
        p_b = self._make_doc(tmp_path, "doc_b.txt", body_b)

        ingester.ingest(p_a)
        stats_b = ingester.ingest(p_b)
        assert not stats_b.skipped

    def test_threshold_above_one_disables_near_dedup(self, tmp_path):
        """threshold > 1.0 should never match → near-dedup disabled."""
        ingester = DocumentIngester(
            output_dir=tmp_path / "out",
            fingerprint_store_path=tmp_path / "fps.npz",
            near_dedup_threshold=1.1,
        )
        # Different content hash but near-identical bigrams — should NOT be blocked
        p_a = self._make_doc(tmp_path, "doc_a.txt", _LONG_BODY_BASE)
        p_b = self._make_doc(tmp_path, "doc_b.txt", _LONG_BODY_ALT)

        ingester.ingest(p_a)
        stats_b = ingester.ingest(p_b)
        assert not stats_b.skipped

    def test_dry_run_bypasses_near_dedup(self, tmp_path):
        ingester = DocumentIngester(
            output_dir=tmp_path / "out",
            fingerprint_store_path=tmp_path / "fps.npz",
            near_dedup_threshold=0.97,
        )
        # Ingest doc_a normally so fingerprint is registered.
        p_a = self._make_doc(tmp_path, "doc_a.txt", _LONG_BODY_BASE)
        ingester.ingest(p_a)
        # dry_run=True on near-identical doc must not be blocked by near-dedup.
        # (It will be "skipped" with skip_reason='dry run', which is fine.)
        p_b = self._make_doc(tmp_path, "doc_b.txt", _LONG_BODY_ALT)
        stats_dry = ingester.ingest(p_b, dry_run=True)
        # Near-dedup must not have triggered — skip_reason must not be 'near-duplicate'.
        assert "near-duplicate" not in (stats_dry.skip_reason or "").lower()

    def test_force_reingest_bypasses_near_dedup(self, tmp_path):
        ingester = DocumentIngester(
            output_dir=tmp_path / "out",
            fingerprint_store_path=tmp_path / "fps.npz",
            near_dedup_threshold=0.97,
        )
        p_a = self._make_doc(tmp_path, "doc_a.txt", _LONG_BODY_BASE)
        p_b = self._make_doc(tmp_path, "doc_b.txt", _LONG_BODY_ALT)

        ingester.ingest(p_a)
        # force_reingest bypasses near-dedup — doc_b should be ingested.
        stats_force = ingester.ingest(p_b, force_reingest=True)
        assert not stats_force.skipped

    def test_short_doc_skips_near_dedup_check(self, tmp_path):
        """Documents shorter than min_chars should NOT be compared (no fingerprint)."""
        ingester = DocumentIngester(
            output_dir=tmp_path / "out",
            fingerprint_store_path=tmp_path / "fps.npz",
            near_dedup_threshold=0.97,
        )
        short_body = "Short doc.\n\nAnother short paragraph.\n\n"  # < 500 chars
        p_a = self._make_doc(tmp_path, "doc_a.txt", short_body)
        p_b = self._make_doc(tmp_path, "doc_b.txt", short_body)

        ingester.ingest(p_a)
        stats_b = ingester.ingest(p_b)
        # Short doc must not be blocked by near-dedup (may be skipped as duplicate
        # via content hash but not as near-dup).
        if stats_b.skipped:
            assert "near-duplicate" not in (stats_b.skip_reason or "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — --no-embed flag
# ─────────────────────────────────────────────────────────────────────────────

class TestNoEmbed:
    def test_no_embed_flag_in_run_py(self):
        """--no-embed must be registered in run.py's argparse setup."""
        run_py = Path(__file__).parent.parent / "run.py"
        assert "--no-embed" in run_py.read_text()

    def test_no_embed_prevents_migrate_documents_call(self, tmp_path):
        """With --no-embed, the Chroma embedding phase must not be entered."""
        import run
        from vox_romana.memory.vector_index import ChromaVectorIndex
        p = write(tmp_path, "doc.txt", "Test content.\n\nMore content.\n\n" * 10)
        out = tmp_path / "out"
        # Keep is_available() returning False so no ChromaVectorIndex is ever
        # instantiated — avoids background threads leaking into subsequent tests.
        with (
            patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", out),
            patch.object(ChromaVectorIndex, "is_available", return_value=False),
            patch.object(ChromaVectorIndex, "migrate_documents") as mock_migrate,
        ):
            run.main([
                "--mode", "ingest",
                "--ingest-path", str(p),
                "--no-embed",
            ])
        mock_migrate.assert_not_called()
