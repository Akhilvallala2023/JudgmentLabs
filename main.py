#!/usr/bin/env python3
"""
Job Matcher - Main Entry Point

This script provides a command-line interface for matching resumes with job listings.
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import application modules
from src.resume_parser import parse_resume
from src.job_fetcher import fetch_jobs
from src.matcher import match_resume_to_jobs
from src.utils import (
    setup_directories, 
    load_config, 
    save_results, 
    print_results,
    validate_resume_path
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Match your resume with remote job listings.'
    )
    
    parser.add_argument(
        '--resume', 
        type=str, 
        help='Path to your resume file (PDF or DOCX)'
    )
    
    parser.add_argument(
        '--feed', 
        type=str, 
        help='URL of the job RSS feed'
    )
    
    parser.add_argument(
        '--jobs', 
        type=int, 
        default=5,
        help='Number of top job matches to display'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        help='Name of the Sentence Transformer model to use'
    )
    
    parser.add_argument(
        '--save', 
        action='store_true',
        help='Save the results to a file'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        help='Path to save the results (JSON format)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='.env',
        help='Path to the configuration file'
    )
    
    return parser.parse_args()

def run_job_matcher(args: argparse.Namespace) -> Optional[List[Dict[str, Any]]]:
    """
    Run the job matcher with the provided arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of job matches or None if an error occurred
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.resume:
        config['resume_path'] = args.resume
    if args.feed:
        config['rss_feed_url'] = args.feed
    if args.jobs:
        config['top_n'] = args.jobs
    if args.model:
        config['model_name'] = args.model
        
    # Set up directories
    setup_directories(config['data_dir'])
    
    # Validate resume path
    resume_path = config['resume_path']
    if not validate_resume_path(resume_path):
        logger.error("Please provide a valid resume file.")
        return None
    
    # Parse resume
    logger.info(f"Parsing resume from {resume_path}")
    resume_data = parse_resume(resume_path)
    if not resume_data:
        logger.error("Failed to parse resume.")
        return None
    
    resume_text = resume_data['text']
    logger.info(f"Successfully parsed resume ({resume_data['word_count']} words)")
    
    # Fetch jobs
    logger.info(f"Fetching jobs from {config['rss_feed_url']}")
    jobs = fetch_jobs(
        config['rss_feed_url'], 
        config['max_jobs'], 
        os.path.join(config['data_dir'], 'cache')
    )
    
    if not jobs:
        logger.error("No jobs found.")
        return None
    
    logger.info(f"Fetched {len(jobs)} jobs")
    
    # Match resume to jobs
    logger.info("Matching resume to jobs...")
    matches = match_resume_to_jobs(
        resume_text, 
        jobs, 
        config['top_n'], 
        config['model_name']
    )
    
    return matches

def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Run the job matcher
        matches = run_job_matcher(args)
        
        if not matches:
            return 1
        
        # Print results
        print_results(matches)
        
        # Save results if requested
        if args.save or args.output:
            output_path = args.output
            save_results(matches, output_path)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Job matching interrupted by user.")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 