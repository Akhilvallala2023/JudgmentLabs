"""
Matcher Module

This module provides functionality to match resumes with job listings
using semantic similarity with Sentence Transformers.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import re
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers, but provide fallback if not available
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Using sentence-transformers for embeddings")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using simple fallback embedding method")

class SimpleEmbedder:
    """
    A simple fallback embedding class when sentence-transformers is not available.
    Uses TF-IDF like approach for creating document vectors.
    """
    
    def __init__(self):
        """Initialize the simple embedder."""
        self.vocab = {}  # word -> index mapping
        self.idf = {}    # word -> inverse document frequency
        self.documents = []
        self.vector_size = 100  # Fixed vector size for consistent dimensions
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization function.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric characters
        return re.findall(r'\b[a-z][a-z0-9]*\b', text.lower())
    
    def _update_vocab(self, tokens: List[str]) -> None:
        """
        Update vocabulary with new tokens.
        
        Args:
            tokens: List of tokens
        """
        for token in set(tokens):  # Use set to count each word once per document
            if token not in self.vocab and len(self.vocab) < self.vector_size:
                self.vocab[token] = len(self.vocab)
            
            # Update document frequency
            self.idf[token] = self.idf.get(token, 0) + 1
    
    def _compute_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Compute TF-IDF vector for a document.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Document vector
        """
        # Create a zero vector with fixed size
        vector = np.zeros(self.vector_size)
        
        # Count term frequencies
        term_freq = {}
        for token in tokens:
            if token in self.vocab:
                term_freq[token] = term_freq.get(token, 0) + 1
        
        # Compute TF-IDF
        doc_count = len(self.documents) + 1  # Add 1 for current document
        for token, freq in term_freq.items():
            if token in self.vocab:  # Check if token is in vocab (might not be if vocab is full)
                idx = self.vocab[token]
                tf = freq / max(1, len(tokens))
                idf = np.log(max(1, doc_count) / max(1, self.idf.get(token, 0)))
                vector[idx] = tf * idf
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def encode(self, sentences: List[str], convert_to_tensor: bool = False) -> np.ndarray:
        """
        Encode sentences into vectors.
        
        Args:
            sentences: List of sentences to encode
            convert_to_tensor: Ignored, kept for API compatibility
            
        Returns:
            Array of document vectors
        """
        # First pass: build vocabulary from all sentences
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            self._update_vocab(tokens)
            
        # Second pass: compute vectors using the complete vocabulary
        vectors = []
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            vector = self._compute_vector(tokens)
            vectors.append(vector)
            
        # Add to documents for future IDF calculations
        self.documents.extend(sentences)
        
        return np.array(vectors)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    try:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

class JobMatcher:
    """Class for matching resumes with job listings using semantic similarity."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the JobMatcher.
        
        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        logger.info(f"Initializing JobMatcher with model: {model_name}")
        self.model_name = model_name
        
        # Initialize the model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer model: {e}")
                logger.warning("Falling back to simple embedder")
                self.model = SimpleEmbedder()
        else:
            self.model = SimpleEmbedder()
        
        # Common stop words to filter out
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'with', 'by', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
    
    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better embedding.
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum number of words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def create_embeddings(self, text_chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            text_chunks: List of text chunks to embed
            
        Returns:
            Array of embeddings
        """
        return self.model.encode(text_chunks, convert_to_tensor=True)
    
    def extract_key_terms(self, text: str, max_terms: int = 5) -> List[str]:
        """
        Extract key terms from text using frequency analysis.
        
        Args:
            text: Text to extract terms from
            max_terms: Maximum number of terms to extract
            
        Returns:
            List of key terms
        """
        # Extract words and clean them
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#-]{2,}\b', text.lower())
        words = [w for w in words if w not in self.stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get the most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in sorted_words[:max_terms]]
        
        return top_words
    
    def match_resume_to_job(
        self, 
        resume_text: str, 
        job_title: str, 
        job_description: str
    ) -> Dict[str, Any]:
        """
        Match a resume to a job using semantic similarity.
        
        Args:
            resume_text: Text content of the resume
            job_title: Title of the job
            job_description: Description of the job
            
        Returns:
            Dictionary with match results
        """
        try:
            # Prepare job information
            job_info = f"Job Title: {job_title}\n\nJob Description: {job_description}"
            
            # Chunk the resume and job description
            resume_chunks = self.chunk_text(resume_text)
            job_chunks = self.chunk_text(job_info)
            
            # Create embeddings
            resume_embeddings = self.create_embeddings(resume_chunks)
            job_embeddings = self.create_embeddings(job_chunks)
            
            # Calculate cosine similarities between all chunks
            similarities = []
            chunk_similarities = []
            
            for i, resume_emb in enumerate(resume_embeddings):
                for j, job_emb in enumerate(job_embeddings):
                    if SENTENCE_TRANSFORMERS_AVAILABLE:
                        try:
                            similarity = util.pytorch_cos_sim(resume_emb, job_emb).item()
                        except:
                            similarity = cosine_similarity(resume_emb, job_emb)
                    else:
                        similarity = cosine_similarity(resume_emb, job_emb)
                        
                    similarities.append(similarity)
                    chunk_similarities.append((similarity, i, j))
            
            # Calculate overall match score (0-100)
            # Using the average of top 5 similarities or all if less than 5
            top_similarities = sorted(similarities, reverse=True)[:min(5, len(similarities))]
            avg_similarity = sum(top_similarities) / len(top_similarities) if top_similarities else 0
            score = int(avg_similarity * 100)
            
            # Identify strengths and gaps
            strengths = []
            gaps = []
            
            # Sort chunk similarities by score (descending)
            chunk_similarities.sort(reverse=True)
            
            # Extract strengths from top matches
            for similarity, resume_idx, _ in chunk_similarities[:3]:
                if resume_idx < len(resume_chunks):
                    key_terms = self.extract_key_terms(resume_chunks[resume_idx])
                    if key_terms:
                        strengths.extend(key_terms)
            
            # Extract gaps from job chunks with lowest similarity
            bottom_matches = sorted(chunk_similarities, key=lambda x: x[0])[:3]
            for _, _, job_idx in bottom_matches:
                if job_idx < len(job_chunks):
                    key_terms = self.extract_key_terms(job_chunks[job_idx])
                    if key_terms:
                        gaps.extend(key_terms)
            
            # Remove duplicates while preserving order
            strengths = list(dict.fromkeys(strengths))[:5]
            gaps = list(dict.fromkeys(gaps))[:5]
            
            # Prepare the match result
            match_result = {
                'score': score,
                'strengths': strengths,
                'gaps': gaps,
                'job_title': job_title,
                'similarity_details': {
                    'max': max(similarities) if similarities else 0,
                    'min': min(similarities) if similarities else 0,
                    'avg': avg_similarity,
                    'count': len(similarities)
                }
            }
            
            return match_result
        
        except Exception as e:
            logger.error(f"Error in match_resume_to_job: {e}")
            # Return a default match result
            return {
                'score': 0,
                'strengths': [],
                'gaps': [],
                'job_title': job_title,
                'similarity_details': {
                    'max': 0,
                    'min': 0,
                    'avg': 0,
                    'count': 0
                }
            }
    
    def find_matching_jobs(
        self, 
        resume_text: str, 
        jobs: List[Dict[str, Any]], 
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the top matching jobs for a resume.
        
        Args:
            resume_text: Text content of the resume
            jobs: List of job dictionaries to match against
            top_n: Number of top matches to return
            
        Returns:
            List of job matches with scores and analysis
        """
        logger.info(f"Finding top {top_n} matches among {len(jobs)} jobs")
        
        job_matches = []
        
        # Process each job with a progress bar
        for job in tqdm(jobs, desc="Matching jobs"):
            try:
                # Match the resume to the job
                match_result = self.match_resume_to_job(
                    resume_text,
                    job.get('title', ''),
                    job.get('description', '')
                )
                
                # Add the match result to the job
                job_match = {
                    'job': job,
                    'score': match_result['score'],
                    'strengths': match_result['strengths'],
                    'gaps': match_result['gaps'],
                    'similarity_details': match_result['similarity_details']
                }
                
                job_matches.append(job_match)
                
            except Exception as e:
                logger.error(f"Error matching job {job.get('id', 'unknown')}: {e}")
        
        # Sort by score (descending)
        job_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top matches
        return job_matches[:top_n]


def match_resume_to_jobs(
    resume_text: str, 
    jobs: List[Dict[str, Any]], 
    top_n: int = 5,
    model_name: str = 'all-MiniLM-L6-v2'
) -> List[Dict[str, Any]]:
    """
    Convenience function to match a resume to multiple jobs.
    
    Args:
        resume_text: Text content of the resume
        jobs: List of job dictionaries to match against
        top_n: Number of top matches to return
        model_name: Name of the Sentence Transformer model to use
        
    Returns:
        List of job matches with scores and analysis
    """
    matcher = JobMatcher(model_name)
    return matcher.find_matching_jobs(resume_text, jobs, top_n) 