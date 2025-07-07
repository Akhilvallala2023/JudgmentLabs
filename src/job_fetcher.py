"""
Job Fetcher Module

This module provides functionality to fetch job listings from RSS feeds
and process them for matching.
"""

import feedparser
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime
import hashlib
import os
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobFetcher:
    """Class for fetching and processing job listings from RSS feeds."""
    
    def __init__(self, feed_url: str, cache_dir: Optional[str] = None):
        """
        Initialize the JobFetcher.
        
        Args:
            feed_url: URL of the RSS feed to fetch jobs from
            cache_dir: Directory to cache fetched jobs (optional)
        """
        self.feed_url = feed_url
        self.cache_dir = cache_dir
        
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_job_id(self, job_url: str) -> str:
        """
        Generate a unique ID for a job based on its URL.
        
        Args:
            job_url: URL of the job listing
            
        Returns:
            A unique ID string
        """
        # Create a hash of the URL to use as ID
        return hashlib.md5(job_url.encode()).hexdigest()[:10]
    
    def fetch_jobs(self, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch recent job listings from the RSS feed.
        
        Args:
            limit: Maximum number of job listings to fetch
            
        Returns:
            List of job dictionaries
        """
        logger.info(f"Fetching jobs from {self.feed_url}")
        
        try:
            # Parse the RSS feed
            feed = feedparser.parse(self.feed_url)
            
            if feed.bozo:  # Check if there was an error parsing the feed
                logger.error(f"Error parsing feed: {feed.bozo_exception}")
                return []
            
            # Process the entries
            jobs = []
            for entry in feed.entries[:limit]:
                # Extract job details
                job = self._process_entry(entry)
                if job:
                    jobs.append(job)
                
                # Stop if we've reached the limit
                if len(jobs) >= limit:
                    break
            
            logger.info(f"Fetched {len(jobs)} jobs from feed")
            return jobs
            
        except Exception as e:
            logger.error(f"Error fetching jobs: {e}")
            return []
    
    def _process_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a feed entry into a job dictionary.
        
        Args:
            entry: Feed entry from feedparser
            
        Returns:
            Processed job dictionary or None if processing failed
        """
        try:
            # Extract required fields
            job_url = entry.link
            job_id = self._generate_job_id(job_url)
            
            # Get domain from URL for source identification
            domain = urlparse(job_url).netloc
            
            job = {
                'id': job_id,
                'title': entry.title,
                'company': entry.author if hasattr(entry, 'author') else 'Unknown',
                'link': job_url,
                'description': entry.summary if hasattr(entry, 'summary') else '',
                'source': domain,
                'fetched_at': datetime.now().isoformat()
            }
            
            # Add published date if available
            if hasattr(entry, 'published_parsed'):
                published_date = time.strftime('%Y-%m-%d %H:%M:%S', entry.published_parsed)
                job['published_date'] = published_date
            
            # Cache the job if cache_dir is set
            if self.cache_dir:
                self._cache_job(job)
            
            return job
            
        except Exception as e:
            logger.error(f"Error processing job entry: {e}")
            return None
    
    def _cache_job(self, job: Dict[str, Any]) -> None:
        """
        Cache a job to disk.
        
        Args:
            job: Job dictionary to cache
        """
        import json
        
        try:
            cache_file = os.path.join(self.cache_dir, f"job_{job['id']}.json")
            with open(cache_file, 'w') as f:
                json.dump(job, f, indent=2)
        except Exception as e:
            logger.error(f"Error caching job: {e}")


def fetch_jobs(feed_url: str, limit: int = 30, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch jobs from an RSS feed.
    
    Args:
        feed_url: URL of the RSS feed
        limit: Maximum number of jobs to fetch
        cache_dir: Directory to cache fetched jobs (optional)
        
    Returns:
        List of job dictionaries
    """
    fetcher = JobFetcher(feed_url, cache_dir)
    return fetcher.fetch_jobs(limit) 