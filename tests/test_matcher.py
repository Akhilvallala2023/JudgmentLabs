"""
Tests for the matcher module.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the module being tested
from src.matcher import JobMatcher, match_resume_to_jobs, SimpleEmbedder

class TestJobMatcher(unittest.TestCase):
    """Test cases for the JobMatcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a matcher instance
        self.matcher = JobMatcher('mock-model')
        
        # Replace the model with a mock
        self.mock_embedder = MagicMock(spec=SimpleEmbedder)
        self.matcher.model = self.mock_embedder
        
        # Configure the mock to return consistent embeddings
        self.mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    
    def test_chunk_text(self):
        """Test chunking text into smaller pieces."""
        # Create a test text with 10 words
        text = "This is a test text with exactly ten words."
        
        # Chunk with size 3 and overlap 1
        chunks = self.matcher.chunk_text(text, chunk_size=3, overlap=1)
        
        # Check the results - actual implementation returns 4 chunks
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], "This is a")
        self.assertEqual(chunks[1], "a test text")
        self.assertEqual(chunks[2], "text with exactly")
        self.assertEqual(chunks[3], "exactly ten words.")
    
    def test_extract_key_terms(self):
        """Test extracting key terms from text."""
        text = "Python developer with experience in machine learning and data science. Python is required."
        
        # Extract top 3 terms
        terms = self.matcher.extract_key_terms(text, max_terms=3)
        
        # Check the results
        self.assertEqual(len(terms), 3)
        self.assertIn("python", terms)
        self.assertIn("developer", terms)
        self.assertIn("experience", terms)
    
    def test_match_resume_to_job(self):
        """Test matching a resume to a job."""
        # Create test data
        resume_text = "Python developer with 5 years of experience"
        job_title = "Senior Python Developer"
        job_description = "Looking for a Python developer with experience"
        
        # Configure the mock to return specific values
        self.mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Call the method
        result = self.matcher.match_resume_to_job(resume_text, job_title, job_description)
        
        # Check the results
        self.assertIsInstance(result, dict)
        self.assertIn('score', result)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 100)
        self.assertIn('strengths', result)
        self.assertIn('gaps', result)
        self.assertEqual(result['job_title'], job_title)
    
    @patch('src.matcher.JobMatcher.match_resume_to_job')
    def test_find_matching_jobs(self, mock_match):
        """Test finding matching jobs for a resume."""
        # Mock the match_resume_to_job method
        mock_match.side_effect = [
            {'score': 90, 'strengths': ['python'], 'gaps': ['java'], 'job_title': 'Job 1', 'similarity_details': {}},
            {'score': 70, 'strengths': ['python'], 'gaps': ['java'], 'job_title': 'Job 2', 'similarity_details': {}},
            {'score': 80, 'strengths': ['python'], 'gaps': ['java'], 'job_title': 'Job 3', 'similarity_details': {}}
        ]
        
        # Create test data
        resume_text = "Python developer"
        jobs = [
            {'id': '1', 'title': 'Job 1', 'description': 'Desc 1'},
            {'id': '2', 'title': 'Job 2', 'description': 'Desc 2'},
            {'id': '3', 'title': 'Job 3', 'description': 'Desc 3'}
        ]
        
        # Call the method
        matches = self.matcher.find_matching_jobs(resume_text, jobs, top_n=2)
        
        # Check the results
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0]['score'], 90)
        self.assertEqual(matches[0]['job']['id'], '1')
        self.assertEqual(matches[1]['score'], 80)
        self.assertEqual(matches[1]['job']['id'], '3')
    
    @patch('src.matcher.JobMatcher')
    def test_convenience_function(self, MockJobMatcher):
        """Test the convenience function for matching jobs."""
        # Configure the mock
        mock_instance = MockJobMatcher.return_value
        mock_instance.find_matching_jobs.return_value = [{'job': {'title': 'Test Job'}, 'score': 90}]
        
        # Call the function
        resume_text = "Python developer"
        jobs = [{'title': 'Test Job', 'description': 'Test description'}]
        matches = match_resume_to_jobs(resume_text, jobs, top_n=1)
        
        # Check the results
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['job']['title'], 'Test Job')
        self.assertEqual(matches[0]['score'], 90)
        mock_instance.find_matching_jobs.assert_called_once()

if __name__ == '__main__':
    unittest.main() 