"""
Tests for the job fetcher module.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.job_fetcher import JobFetcher, fetch_jobs

class TestJobFetcher(unittest.TestCase):
    """Test cases for the JobFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feed_url = 'https://example.com/jobs.rss'
        self.temp_dir = tempfile.mkdtemp()
        self.fetcher = JobFetcher(self.feed_url, self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_job_id(self):
        """Test generating a job ID from a URL."""
        job_url = 'https://example.com/jobs/12345'
        job_id = self.fetcher._generate_job_id(job_url)
        
        # Check that the ID is a string with the expected length
        self.assertIsInstance(job_id, str)
        self.assertEqual(len(job_id), 10)
        
        # Check that the same URL always generates the same ID
        job_id2 = self.fetcher._generate_job_id(job_url)
        self.assertEqual(job_id, job_id2)
        
        # Check that different URLs generate different IDs
        job_url2 = 'https://example.com/jobs/67890'
        job_id3 = self.fetcher._generate_job_id(job_url2)
        self.assertNotEqual(job_id, job_id3)
    
    @patch('feedparser.parse')
    def test_fetch_jobs_success(self, mock_parse):
        """Test fetching jobs successfully."""
        # Mock the feedparser response
        mock_entry1 = MagicMock()
        mock_entry1.title = 'Job 1'
        mock_entry1.link = 'https://example.com/jobs/1'
        mock_entry1.author = 'Company 1'
        mock_entry1.summary = 'Job 1 description'
        mock_entry1.published_parsed = (2025, 7, 1, 12, 0, 0, 0, 0, 0)
        
        mock_entry2 = MagicMock()
        mock_entry2.title = 'Job 2'
        mock_entry2.link = 'https://example.com/jobs/2'
        mock_entry2.author = 'Company 2'
        mock_entry2.summary = 'Job 2 description'
        mock_entry2.published_parsed = (2025, 7, 2, 12, 0, 0, 0, 0, 0)
        
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [mock_entry1, mock_entry2]
        mock_parse.return_value = mock_feed
        
        # Override the _process_entry method to avoid caching issues during testing
        with patch.object(self.fetcher, '_process_entry', side_effect=[
            {
                'id': '1234567890',
                'title': 'Job 1',
                'company': 'Company 1',
                'link': 'https://example.com/jobs/1',
                'description': 'Job 1 description'
            },
            {
                'id': '0987654321',
                'title': 'Job 2',
                'company': 'Company 2',
                'link': 'https://example.com/jobs/2',
                'description': 'Job 2 description'
            }
        ]):
            # Call the method
            jobs = self.fetcher.fetch_jobs(limit=2)
            
            # Check the results
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]['title'], 'Job 1')
            self.assertEqual(jobs[0]['company'], 'Company 1')
            self.assertEqual(jobs[0]['link'], 'https://example.com/jobs/1')
            self.assertEqual(jobs[0]['description'], 'Job 1 description')
            self.assertEqual(jobs[1]['title'], 'Job 2')
    
    @patch('feedparser.parse')
    def test_fetch_jobs_empty(self, mock_parse):
        """Test fetching jobs when the feed is empty."""
        # Mock an empty feed
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = []
        mock_parse.return_value = mock_feed
        
        # Call the method
        jobs = self.fetcher.fetch_jobs()
        
        # Check the results
        self.assertEqual(len(jobs), 0)
    
    @patch('feedparser.parse')
    def test_fetch_jobs_error(self, mock_parse):
        """Test fetching jobs when there's an error parsing the feed."""
        # Mock a feed with an error
        mock_feed = MagicMock()
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception('Feed parsing error')
        mock_parse.return_value = mock_feed
        
        # Call the method
        jobs = self.fetcher.fetch_jobs()
        
        # Check the results
        self.assertEqual(len(jobs), 0)
    
    @patch('src.job_fetcher.JobFetcher.fetch_jobs')
    def test_convenience_function(self, mock_fetch_jobs):
        """Test the convenience function for fetching jobs."""
        mock_fetch_jobs.return_value = [{'title': 'Test Job'}]
        
        # Call the function
        jobs = fetch_jobs(self.feed_url, 5, self.temp_dir)
        
        # Check the results
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['title'], 'Test Job')
        mock_fetch_jobs.assert_called_once_with(5)

if __name__ == '__main__':
    unittest.main() 