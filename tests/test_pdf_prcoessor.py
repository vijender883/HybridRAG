# tests/test_pdf_processor.py

import pytest
from pathlib import Path
from collections import Counter

# Make sure the src directory is in the path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backend.utils.pdf_processor import PDFProcessor, TableInfo, TableSchema

# This is the sample text our "mocked" PDF will return.
# It includes a heading, paragraphs, and bullets to test the sectionize logic.
MOCK_PDF_TEXT = """
INTRODUCTION TO MOCKING

This is the first paragraph. It contains several sentences to test the sentence tokenizer and chunking logic. We want to see if this stays together.

This is a second paragraph, separated by a blank line. The chunker should respect this boundary.

KEY FEATURES
- Feature one is about mocking external calls.
- Feature two is about isolating the function under test.
- Feature three ensures tests are fast and repeatable.

This final sentence concludes the section.
"""

def test_optimized_chunking_with_mocks(mocker):
    """
    Tests the text chunking logic of `optimized_extract_and_store`
    by mocking all external dependencies (PDFs, Gemini, Database).
    """
    # 1. ARRANGE: Set up the environment and mocks

    # Mock the dependencies in the PDFProcessor's __init__ method
    # so we don't need a real database or Gemini API key to create an instance.
    mock_engine = mocker.patch('src.backend.utils.pdf_processor.create_engine')
    mock_configure = mocker.patch('src.backend.utils.pdf_processor.genai.configure')
    mock_model = mocker.patch('src.backend.utils.pdf_processor.genai.GenerativeModel')
    
    # Mock the engine connection
    mock_conn = mocker.Mock()
    mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.return_value.connect.return_value.__exit__.return_value = None

    # Mock file system interactions
    mocker.patch.object(Path, 'exists', return_value=False) # Mocks schema file loading

    # Create an instance of the processor. This is safe now.
    processor = PDFProcessor(database_url="postgresql://user:pass@localhost/db", gemini_api_key="mock_api_key")

    # Mock the pdfplumber library to avoid needing a real PDF file.
    # We create a fake 'page' object that has the methods our function calls.
    mock_page = mocker.Mock()
    mock_page.extract_text.return_value = MOCK_PDF_TEXT
    mock_page.extract_tables.return_value = [] # No tables for this test

    # The 'pdf' object needs to act as a context manager ('with' statement)
    mock_pdf = mocker.MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = mocker.Mock(return_value=mock_pdf)
    mock_pdf.__exit__ = mocker.Mock(return_value=None)
    mocker.patch('src.backend.utils.pdf_processor.pdfplumber.open', return_value=mock_pdf)

    # Mock the helper methods that would be called for tables.
    # We don't expect them to be called since extract_tables returns [],
    # but it's good practice to mock them anyway.
    mocker.patch.object(processor, '_store_table_with_schema', return_value=True)

    # 2. ACT: Call the function we want to test
    # (Assuming you've added your `optimized_extract_and_store` method to the class)
    result = processor.optimized_extract_and_store(pdf_path="fake/path/to/doc.pdf")

    # 3. ASSERT: Check if the output is what we expect
    assert "text_chunks" in result
    assert isinstance(result["text_chunks"], list)
    assert len(result["text_chunks"]) > 0 # We expect at least one chunk

    # Check the content of the first chunk
    first_chunk = result["text_chunks"][0]
    assert "INTRODUCTION TO MOCKING — (Page 1)" in first_chunk
    assert "This is the first paragraph." in first_chunk
    
    # Check that we have multiple chunks (the chunking algorithm is working)
    assert len(result["text_chunks"]) >= 2, f"Expected at least 2 chunks, got {len(result['text_chunks'])}"
    
    # Check that the content is properly distributed across chunks
    all_content = " ".join(result["text_chunks"])
    assert "This is a second paragraph" in all_content
    assert "KEY FEATURES" in all_content
    assert "- Feature one is about mocking" in all_content

    print("\n✅ Test passed: Text was chunked as expected.")
    for i, chunk in enumerate(result['text_chunks']):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)