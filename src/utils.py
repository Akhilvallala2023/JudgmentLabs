"""
Utilities Module

This module provides utility functions for the job matcher application.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(base_dir: str = 'data') -> Dict[str, str]:
    """
    Set up the directory structure for the application.
    
    Args:
        base_dir: Base directory for data storage
        
    Returns:
        Dictionary of directory paths
    """
    directories = {
        'base': base_dir,
        'cache': os.path.join(base_dir, 'cache'),
        'jobs': os.path.join(base_dir, 'jobs'),
        'resumes': os.path.join(base_dir, 'resumes'),
        'results': os.path.join(base_dir, 'results')
    }
    
    # Create directories if they don't exist
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return directories

def load_config(config_path: str = '.env') -> Dict[str, Any]:
    """
    Load configuration from a .env file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary of configuration values
    """
    import dotenv
    
    # Load environment variables from .env file
    dotenv.load_dotenv(config_path)
    
    # Get configuration values
    config = {
        'rss_feed_url': os.getenv('RSS_FEED_URL', 'https://remoteok.com/remote-jobs.rss'),
        'max_jobs': int(os.getenv('MAX_JOBS', '30')),
        'model_name': os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2'),
        'resume_path': os.getenv('RESUME_PATH', ''),
        'data_dir': os.getenv('DATA_DIR', 'data'),
        'top_n': int(os.getenv('TOP_N', '5'))
    }
    
    return config

def save_results(
    matches: List[Dict[str, Any]], 
    output_path: Optional[str] = None,
    include_timestamp: bool = True
) -> str:
    """
    Save match results to a JSON file.
    
    Args:
        matches: List of job matches
        output_path: Path to save the results (optional)
        include_timestamp: Whether to include a timestamp in the filename
        
    Returns:
        Path to the saved results file
    """
    # Generate output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
        filename = f"job_matches_{timestamp}.json" if include_timestamp else "job_matches.json"
        output_path = os.path.join('data', 'results', filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare the results for saving
    results = {
        'timestamp': datetime.now().isoformat(),
        'match_count': len(matches),
        'matches': matches
    }
    
    # Save the results
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    return output_path

def format_match_for_display(match: Dict[str, Any]) -> str:
    """
    Format a job match for display.
    
    Args:
        match: Job match dictionary
        
    Returns:
        Formatted string representation of the match
    """
    job = match['job']
    
    # Format strengths and gaps as bullet points
    strengths_text = "\n".join([f"• {s}" for s in match.get('strengths', [])])
    gaps_text = "\n".join([f"• {s}" for s in match.get('gaps', [])])
    
    # Build the formatted text
    formatted = f"""
Match Score: {match['score']}/100

Job Title: {job['title']}
Company: {job.get('company', 'Unknown')}
Link: {job['link']}

STRENGTHS:
{strengths_text if strengths_text else "None identified"}

GAPS:
{gaps_text if gaps_text else "None identified"}
"""
    
    return formatted.strip()

def print_results(matches: List[Dict[str, Any]]) -> None:
    """
    Print job match results to the console.
    
    Args:
        matches: List of job matches
    """
    if not matches:
        print("No matching jobs found.")
        return
    
    print(f"\n===== TOP {len(matches)} JOB MATCHES FOR YOUR RESUME =====\n")
    
    for idx, match in enumerate(matches, 1):
        print(f"Match #{idx}")
        print(format_match_for_display(match))
        print("-" * 70)

def validate_resume_path(resume_path: str) -> bool:
    """
    Validate that a resume file exists and is of a supported format.
    
    Args:
        resume_path: Path to the resume file
        
    Returns:
        True if valid, False otherwise
    """
    if not resume_path:
        logger.error("Resume path is empty")
        return False
    
    if not os.path.exists(resume_path):
        logger.error(f"Resume file not found: {resume_path}")
        return False
    
    file_ext = os.path.splitext(resume_path)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        logger.error(f"Unsupported resume format: {file_ext}")
        return False
    
    return True 