from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from datetime import datetime
from ..core import logger

@dataclass
class PromptTemplate:
    """Template for structuring prompts."""
    template: str
    context_prefix: str = "Based on the following context:"
    version: str = "1.0"
    created_at: str = datetime.now().isoformat()
    
    def format(self, context: str, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format the prompt template with provided context and query."""
        if metadata is None:
            metadata = {}
            
        return f"{self.context_prefix}\n\n{context}\n\n{self.template.format(
            query=query,
            **metadata
        )}"
    
    def get_signature(self) -> str:
        """Generate unique signature for template."""
        content = f"{self.template}{self.context_prefix}{self.version}"
        return hashlib.md5(content.encode()).hexdigest()

class PromptHandler:
    """Handles prompt generation and formatting for different query types."""
    
    def __init__(self):
        # Initialize templates for different data types
        self._data_templates: Dict[str, Dict[str, PromptTemplate]] = {
            # Tabular data templates
            'row': {
                'exact': PromptTemplate(
                    template="Looking at row {row_number} of the data:",
                    version="1.0"
                ),
                'ordinal': PromptTemplate(
                    template="Looking at the {ordinal} row of the data:",
                    version="1.0"
                ),
                'general': PromptTemplate(
                    template="Please provide the data from row {row_number}.",
                    version="1.0"
                )
            },
            'pdf': {
                'page': PromptTemplate(
                    template="Looking at page {page_number} of the PDF:",
                    version="1.0"
                ),
                'summary': PromptTemplate(
                    template="Based on the PDF content:",
                    version="1.0"
                )
            },
            'image': {
                'ocr': PromptTemplate(
                    template="Based on the text extracted from the image:",
                    version="1.0"
                ),
                'description': PromptTemplate(
                    template="Looking at the image description:",
                    version="1.0"
                )
            },
            'xml': {
                'structure': PromptTemplate(
                    template="Looking at the XML structure:",
                    version="1.0"
                ),
                'content': PromptTemplate(
                    template="Based on the XML content:",
                    version="1.0"
                )
            },
            'enhanced_search': {
                'general': PromptTemplate(
                    template="Analyzing the following content for: {query}",
                    context_prefix="Based on the content analysis:",
                    version="1.0"
                ),
                'semantic': PromptTemplate(
                    template="Using semantic analysis to understand: {query}",
                    context_prefix="Based on semantic relationships in the content:",
                    version="1.0"
                ),
                'tabular': PromptTemplate(
                    template="Analyzing tabular data for: {query}",
                    context_prefix="Based on structured data analysis:",
                    version="1.0"
                ),
                'pdf_text': PromptTemplate(
                    template="Analyzing PDF document content for: {query}",
                    context_prefix="From the PDF content analysis:",
                    version="1.0"
                ),
                'image_ocr': PromptTemplate(
                    template="Analyzing extracted text and image content for: {query}",
                    context_prefix="Based on image analysis and OCR text:",
                    version="1.0"
                ),
                'xml_content': PromptTemplate(
                    template="Analyzing XML structure and content for: {query}",
                    context_prefix="Based on XML content analysis:",
                    version="1.0"
                ),
                'docx_text': PromptTemplate(
                    template="Analyzing document content for: {query}",
                    context_prefix="From document content analysis:",
                    version="1.0"
                ),
                'cross_document': PromptTemplate(
                    template="Finding relationships across documents for: {query}",
                    context_prefix="Based on cross-document analysis:",
                    version="1.0"
                ),
                'fuzzy': PromptTemplate(
                    template="Finding approximate matches for: {query}",
                    context_prefix="Using fuzzy matching results:",
                    version="1.0"
                )
            }
        }
        
        # Context prefixes for different content types
        self._context_prefixes: Dict[str, str] = {
            'csv_row': "Given this row from the CSV data:",
            'csv_summary': "Based on this CSV summary:",
            'excel_row': "Given this row from the Excel data:",
            'excel_summary': "Based on this Excel summary:",
            'tabular': "Using this tabular data:",
            'pdf_text': "From the PDF document:",
            'pdf_page': "From page {page} of the PDF:",
            'docx_text': "From the Word document:",
            'plain_text': "From the text file:",
            'image_ocr': "From the text extracted from the image:",
            'image_metadata': "Based on the image properties:",
            'xml_content': "From the XML content:",
            'xml_structure': "Looking at the XML structure:",
            'general': "Based on the following information:"
        }
        # Add context prefixes for enhanced search types
        self._context_prefixes.update({
            'enhanced_search': "Using enhanced search capabilities:\n{context_info}",
            'semantic_search': "Based on semantic analysis of the content:",
            'fuzzy_search': "Using approximate matching:",
            'faceted_search': "Analyzing across categories:"
        })
        # Template cache
        self._template_cache: Dict[str, PromptTemplate] = {}
        
        # Custom templates storage
        self._custom_templates: Dict[str, Dict[str, PromptTemplate]] = {}
        
        # Initialize cache with base templates
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize template cache with base templates."""
        for category, templates in self._data_templates.items():
            for name, template in templates.items():
                self._cache_template(f"{category}_{name}", template)
    
    def _cache_template(self, key: str, template: PromptTemplate):
        """Cache a template with its signature."""
        signature = template.get_signature()
        self._template_cache[signature] = template
    
    @lru_cache(maxsize=128)
    def _get_content_prefix(self, content_type: str, metadata_tuple: tuple = None) -> str:
        """Get cached content prefix."""
        metadata = dict(metadata_tuple) if metadata_tuple else None
        prefix = self._context_prefixes.get(content_type, self._context_prefixes['general'])
        if metadata and '{page}' in prefix:
            prefix = prefix.format(page=metadata.get('page', ''))
        return prefix
    
    def detect_content_structure(self, context: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Detect the structure/type of the content.
        Handles all possible content types and their variations.
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Get content type from metadata
            content_type = metadata.get('chunk_type', '').lower()
            
            # First check metadata content type
            if content_type:
                # CSV related types
                if any(t in content_type for t in ['csv_row', 'csv_content', 'csv_summary']):
                    return 'csv_row'
                
                # Excel related types
                if any(t in content_type for t in ['excel_row', 'excel_content', 'xlsx', 'xls']):
                    return 'excel_row'
                
                # PDF related types
                if any(t in content_type for t in ['pdf_text', 'pdf_content', 'pdf_page']):
                    return 'pdf_text'
                
                # Image related types
                if any(t in content_type for t in ['image_ocr', 'image_content', 'image_metadata']):
                    return 'image_ocr'
                
                # XML related types
                if any(t in content_type for t in ['xml_content', 'xml_structure', 'xml_data']):
                    return 'xml_content'
                
                # Word document types
                if any(t in content_type for t in ['docx_text', 'word_document', 'doc_content']):
                    return 'docx_text'
                
                # Plain text types
                if any(t in content_type for t in ['text_content', 'plain_text', 'txt']):
                    return 'plain_text'
            
            # Detect based on content patterns
            # Tabular data patterns
            if any(s in context for s in [
                'Row ', 'Column:', 'csv_row', 'excel_row', 
                'Table:', 'Spreadsheet:', 'Dataset:',
                'cell', 'worksheet', 'workbook'
            ]):
                return 'tabular'
            
            # PDF patterns
            if any(s in context for s in [
                'PDF Page', 'Page ', 'pdf_text',
                'Document page:', 'Page number:', 'PDF document:'
            ]):
                return 'pdf_text'
            
            # Image patterns
            if any(s in context for s in [
                'Image:', 'OCR Text:', 'image_ocr',
                'Picture:', 'Photo:', 'Scanned text:',
                'Image dimensions:', 'Resolution:'
            ]):
                return 'image_ocr'
            
            # XML patterns
            if any(s in context for s in [
                'XML', '<', '>', 'xml_content',
                '<?xml', '</>', 'xmlns:', 'XML structure:'
            ]):
                return 'xml_content'
            
            # Word document patterns
            if any(s in context for s in [
                'Word document:', 'DOCX content:',
                'Document text:', 'Microsoft Word:',
                'Paragraph:', 'Section:'
            ]):
                return 'docx_text'
            
            # Additional checks from metadata
            file_type = metadata.get('file_type', '').lower()
            if file_type:
                if file_type in ['csv', 'xlsx', 'xls']:
                    return 'tabular'
                elif file_type in ['pdf']:
                    return 'pdf_text'
                elif file_type in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                    return 'image_ocr'
                elif file_type in ['xml']:
                    return 'xml_content'
                elif file_type in ['docx', 'doc']:
                    return 'docx_text'
                elif file_type in ['txt']:
                    return 'plain_text'
            
            # Check for enhanced search specific structures
            if metadata.get('semantic_search') or metadata.get('fuzzy_matching'):
                return 'enhanced_search'
            
            logger.info(f"Using general content type for context starting with: {context[:100]}...")
            return 'general'
            
        except Exception as e:
            logger.error(f"Error detecting content structure: {str(e)}")
            return 'general'
    
    def create_prompt(self, 
                     context: str, 
                     query: str, 
                     query_type: str,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create appropriate prompt based on context and query type.
        
        Args:
            context: The context text to use in the prompt
            query: The user's query
            query_type: Type of query ('enhanced_search' or other types)
            metadata: Optional metadata for prompt customization
        """
        try:
            metadata = metadata or {}  # Initialize empty dict if None
            # Handle enhanced search queries
            if query_type == 'enhanced_search':
                return self.create_enhanced_search_prompt(context, query, metadata)
            
            if not context.strip():
                return f"Question: {query}"
            
            # Extract and format identifiers
            identifiers = {}
            if metadata:
                # Extract numeric identifiers
                for key in ['row', 'page', 'ordinal']:
                    if key in metadata:
                        try:
                            identifiers[key] = int(metadata[key])
                        except (ValueError, TypeError):
                            pass
            
            # Detect content structure
            content_type = self.detect_content_structure(context, metadata)
            logger.info(f"Detected content type: {content_type}")
            
            # Get template with identifiers converted to tuple
            template = self._get_template(content_type, query_type, self._dict_to_tuple(identifiers))
            
            # Format metadata for template
            template_metadata = {}
            if metadata:
                # Only include simple types in template metadata
                template_metadata = {
                    k: v for k, v in metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
            
            # Add query-specific metadata
            template_metadata.update({
                'content_type': content_type,
                'query_type': query_type
            })
            # Get context prefix
            context_prefix = self._get_content_prefix(content_type, self._dict_to_tuple(template_metadata))
           # context_prefix = self._get_content_prefix(content_type, template_metadata)
            
            # Create final template with correct prefix
            final_template = PromptTemplate(
                template=template.template,
                context_prefix=context_prefix,
                version=template.version
            )
            
            # Format the prompt
            return final_template.format(
                context=context,
                query=query,
                metadata=template_metadata
            )
        
        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}", exc_info=True)  # Added exc_info=True
            # Fallback to simple prompt
            return f"Based on the following context:\n\n{context}\n\nQuestion: {query}"

    def _dict_to_tuple(self, d: Optional[Dict]) -> tuple:
        """Convert dictionary to hashable tuple."""
        if d is None:
            return tuple()
        return tuple(sorted((k, v) for k, v in d.items()))

    @lru_cache(maxsize=128)
    def _get_template(self, content_type: str, query_type: str, identifiers_tuple: tuple = None) -> PromptTemplate:
        """Get appropriate template based on content and query type."""
        # Convert tuple back to dict for processing
        identifiers = dict(identifiers_tuple) if identifiers_tuple else None
        # Handle row-specific queries
        if 'row' in content_type and identifiers:
            if 'ordinal' in identifiers:
                row_num = identifiers['ordinal']
                return PromptTemplate(
                    template=f"Please show row {row_num} from the data.",
                    context_prefix="Looking at these rows:"
                )
            elif 'row' in identifiers:
                row_num = identifiers['row']
                return PromptTemplate(
                    template=f"Please provide row {row_num}.",
                    context_prefix="From the following data:"
                )
        
        # Get template from cache or create new one
        if content_type in self._data_templates:
            return self._data_templates[content_type].get(
                'general', 
                self._data_templates[content_type]['content']
            )
        
        # Default templates for different content types
        templates = {
            'csv_row': PromptTemplate(
                template="Looking at these CSV rows, {query}",
                context_prefix="Data rows:"
            ),
            'excel_row': PromptTemplate(
                template="From these Excel rows, {query}",
                context_prefix="Excel data:"
            ),
            'pdf_text': PromptTemplate(
                template="Based on this PDF content, {query}",
                context_prefix="PDF text:"
            ),
            'image_ocr': PromptTemplate(
                template="From this image text, {query}",
                context_prefix="OCR text:"
            ),
            'xml_content': PromptTemplate(
                template="Looking at this XML, {query}",
                context_prefix="XML content:"
            )
        }
        
        return templates.get(content_type, PromptTemplate(template="Question: {query}"))
    
    def list_templates(self) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """List all available templates."""
        return {
            'base': {
                category: {name: template.version 
                         for name, template in templates.items()}
                for category, templates in self._data_templates.items()
            },
            'custom': {
                name: list(versions.keys())
                for name, versions in self._custom_templates.items()
            }
        }
    
    def create_enhanced_search_prompt(
            self,
            context: str,
            query: str,
            search_metadata: Optional[Dict[str, Any]] = None
        ) -> str:
            """Create prompt specifically for enhanced search queries with content awareness."""
            try:
                if not context.strip():
                    return f"Enhanced Search Query: {query}"

                search_metadata = search_metadata or {}
                
                # Detect content structure
                content_structure = self.detect_content_structure(context, search_metadata)
                
                # Build context information
                context_info = []
                
                # Add search features information
                if search_metadata.get('expanded_terms'):
                    context_info.append(
                        f"Related terms identified: {', '.join(search_metadata['expanded_terms'])}"
                    )
                
                # Add facet information
                if search_metadata.get('facets'):
                    facets_info = []
                    for facet_type, facet_data in search_metadata['facets'].items():
                        if isinstance(facet_data, dict) and facet_data:
                            top_facets = sorted(facet_data.items(), key=lambda x: x[1], reverse=True)[:3]
                            facets_info.append(f"{facet_type}: {', '.join(f[0] for f in top_facets)}")
                    if facets_info:
                        context_info.append("Key categories found:\n" + "\n".join(facets_info))
                
                # Choose appropriate template based on content and search type
                template_key = 'general'
                
                # Priority 1: Content-specific templates
                if content_structure in ['tabular', 'pdf_text', 'image_ocr', 'xml_content', 'docx_text']:
                    template_key = content_structure
                    
                # Priority 2: Search feature templates
                elif search_metadata.get('fuzzy_matching'):
                    template_key = 'fuzzy'
                elif search_metadata.get('semantic_search'):
                    template_key = 'semantic'
                    
                # Priority 3: Cross-document analysis
                elif len(search_metadata.get('document_types', [])) > 1:
                    template_key = 'cross_document'
                
                # Get template
                template = self._data_templates['enhanced_search'].get(
                    template_key,
                    self._data_templates['enhanced_search']['general']
                )
                
                # Build final context prefix
                context_info_text = "\n".join(context_info) if context_info else "Using comprehensive content analysis"
                
                # Create final template
                final_template = PromptTemplate(
                    template=template.template,
                    context_prefix=f"{template.context_prefix}\n{context_info_text}",
                    version=template.version
                )
                
                # Additional metadata for template
                template_metadata = {
                    'content_type': content_structure,
                    'search_type': template_key,
                    'has_facets': bool(search_metadata.get('facets')),
                    'is_semantic': bool(search_metadata.get('semantic_search')),
                    'is_fuzzy': bool(search_metadata.get('fuzzy_matching'))
                }
                
                # Format the prompt
                return final_template.format(
                    context=context,
                    query=query,
                    metadata=template_metadata
                )
                
            except Exception as e:
                logger.error(f"Error creating enhanced search prompt: {str(e)}", exc_info=True)
                return f"Enhanced Search Query: {query}\n\nContext:\n{context}"