from typing import List, Dict, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from datetime import datetime
from ..core import logger

class SearchEnhancer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize search enhancer with required models and resources"""
        self.encoder = SentenceTransformer(model_name)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        # Download required NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Error downloading NLTK resources: {str(e)}")
            self.stop_words = set()

    async def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand the query using multiple techniques:
        1. Key terms extraction
        2. Related terms based on word embeddings
        """
        try:
            # Tokenize and remove stop words
            tokens = nltk.word_tokenize(query.lower())
            tokens = [t for t in tokens if t not in self.stop_words]
            
            expanded_terms = set()
            
            # Add related terms using word embeddings
            if len(tokens) > 0:
                query_embedding = self.encoder.encode([" ".join(tokens)])[0]
                similar_terms = self._find_similar_terms(query_embedding, tokens)
                expanded_terms.update(similar_terms)
            
            # Remove original query terms and clean up
            expanded_terms = expanded_terms - set(tokens)
            expanded_terms = {term.replace('_', ' ') for term in expanded_terms}
            
            # Build expanded query
            expansion_list = list(expanded_terms)[:5]  # Limit to top 5 terms
            expanded_query = f"{query} {' '.join(expansion_list)}"
            
            return expanded_query, expansion_list
            
        except Exception as e:
            logger.error(f"Query expansion error: {str(e)}")
            return query, []

    def _find_similar_terms(self, query_embedding: np.ndarray, original_terms: List[str], 
                          n_terms: int = 3) -> Set[str]:
        """Find similar terms using word embeddings"""
        try:
            # For now, return empty set as this requires a pre-computed vocabulary
            # This can be enhanced later with domain-specific terminology
            return set()
            
        except Exception as e:
            logger.error(f"Error finding similar terms: {str(e)}")
            return set()

    async def rerank_results(self, query: str, results: List[Dict], 
                           top_k: int = 10) -> List[Dict]:
        """
        Rerank search results using semantic similarity
        """
        try:
            if not results:
                return results
                
            # Encode query and documents
            query_embedding = self.encoder.encode([query])[0]
            
            # Calculate semantic similarity scores
            for result in results:
                if 'embedding' in result:
                    doc_embedding = result['embedding']
                    semantic_score = np.dot(query_embedding, doc_embedding)
                    
                    # Combine with original score (weighted average)
                    original_score = result.get('score', 0)
                    result['score'] = 0.7 * semantic_score + 0.3 * original_score
            
            # Sort by combined score
            reranked_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error reranking results: {str(e)}")
            return results

    async def generate_highlights(self, query: str, text: str, 
                                window_size: int = 50) -> List[Dict[str, str]]:
        """
        Generate highlighted snippets from the text based on query terms
        """
        try:
            highlights = []
            query_terms = set(nltk.word_tokenize(query.lower())) - self.stop_words
            
            # Tokenize the text
            text_lower = text.lower()
            
            # Find matches and their contexts
            for term in query_terms:
                start = 0
                while True:
                    pos = text_lower.find(term, start)
                    if pos == -1:
                        break
                        
                    # Get context window
                    context_start = max(0, pos - window_size)
                    context_end = min(len(text), pos + len(term) + window_size)
                    
                    highlight = {
                        'text': text[context_start:context_end],
                        'position': pos,
                        'term': term
                    }
                    highlights.append(highlight)
                    
                    start = pos + len(term)
            
            # Sort highlights by position and remove overlaps
            highlights.sort(key=lambda x: x['position'])
            non_overlapping = self._remove_overlapping_highlights(highlights)
            
            return non_overlapping[:3]  # Return top 3 highlights
            
        except Exception as e:
            logger.error(f"Error generating highlights: {str(e)}")
            return []

    def _remove_overlapping_highlights(self, highlights: List[Dict]) -> List[Dict]:
        """Remove overlapping highlight segments"""
        if not highlights:
            return []
            
        result = [highlights[0]]
        for curr in highlights[1:]:
            prev = result[-1]
            if curr['position'] > prev['position'] + len(prev['text']):
                result.append(curr)
        return result

    async def apply_fuzzy_matching(self, query: str, texts: List[str], 
                                 threshold: float = 0.8) -> List[int]:
        """
        Apply fuzzy matching to find approximate matches
        """
        try:
            from rapidfuzz import fuzz
            
            matches = []
            query_tokens = set(nltk.word_tokenize(query.lower()))
            
            for idx, text in enumerate(texts):
                text_tokens = set(nltk.word_tokenize(text.lower()))
                
                # Calculate token set ratio
                similarity = fuzz.token_set_ratio(query_tokens, text_tokens) / 100.0
                
                if similarity >= threshold:
                    matches.append(idx)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {str(e)}")
            return []