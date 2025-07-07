"""
Tests for the resume parser module.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.resume_parser import ResumeParser, parse_resume

class TestResumeParser(unittest.TestCase):
    """Test cases for the ResumeParser class."""
    
    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        result = parse_resume('/path/to/nonexistent/file.pdf')
        self.assertIsNone(result)
    
    def test_parse_unsupported_format(self):
        """Test parsing a file with an unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            result = parse_resume(temp_file.name)
            self.assertIsNone(result)
    
    @patch('PyPDF2.PdfReader')
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        """Test extracting text from a PDF file."""
        # Mock the PDF reader
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            # Call the method
            result = ResumeParser.extract_text_from_pdf(temp_file.name)
            
            # Check the result
            self.assertEqual(result, "Test PDF content")
            mock_pdf_reader.assert_called_once()
            mock_page.extract_text.assert_called_once()
    
    @patch('docx.Document')
    def test_extract_text_from_docx(self, mock_document):
        """Test extracting text from a DOCX file."""
        # Mock the Document class
        mock_para1 = MagicMock()
        mock_para1.text = "Test DOCX content"
        mock_para2 = MagicMock()
        mock_para2.text = "Second paragraph"
        
        mock_document.return_value.paragraphs = [mock_para1, mock_para2]
        mock_document.return_value.tables = []
        
        with tempfile.NamedTemporaryFile(suffix='.docx') as temp_file:
            # Call the method
            result = ResumeParser.extract_text_from_docx(temp_file.name)
            
            # Check the result
            self.assertEqual(result, "Test DOCX content\nSecond paragraph")
            mock_document.assert_called_once()
    
    @patch('src.resume_parser.ResumeParser.extract_text_from_pdf')
    def test_parse_pdf(self, mock_extract_pdf):
        """Test parsing a PDF file."""
        mock_extract_pdf.return_value = "Test PDF content"
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            # Call the method
            result = parse_resume(temp_file.name)
            
            # Check the result
            self.assertIsNotNone(result)
            self.assertEqual(result['text'], "Test PDF content")
            self.assertEqual(result['file_type'], "pdf")
            self.assertEqual(result['word_count'], 3)
            mock_extract_pdf.assert_called_once_with(temp_file.name)
    
    @patch('src.resume_parser.ResumeParser.extract_text_from_docx')
    def test_parse_docx(self, mock_extract_docx):
        """Test parsing a DOCX file."""
        mock_extract_docx.return_value = "Test DOCX content"
        
        with tempfile.NamedTemporaryFile(suffix='.docx') as temp_file:
            # Call the method
            result = parse_resume(temp_file.name)
            
            # Check the result
            self.assertIsNotNone(result)
            self.assertEqual(result['text'], "Test DOCX content")
            self.assertEqual(result['file_type'], "docx")
            self.assertEqual(result['word_count'], 3)
            mock_extract_docx.assert_called_once_with(temp_file.name)
    
    @patch('src.resume_parser.ResumeParser.extract_text_from_pdf')
    def test_parse_empty_content(self, mock_extract_pdf):
        """Test parsing a file with no content."""
        mock_extract_pdf.return_value = ""
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            # Call the method
            result = parse_resume(temp_file.name)
            
            # Check the result
            self.assertIsNone(result)
            mock_extract_pdf.assert_called_once_with(temp_file.name)

if __name__ == '__main__':
    unittest.main() 