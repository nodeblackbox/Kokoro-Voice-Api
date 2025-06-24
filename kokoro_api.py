import os
import torch
from kokoro import KModel, KPipeline
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
import tempfile
import logging
import io
import html
import unicodedata
from typing import Dict, Optional, Tuple, List, Any, Union
from functools import wraps
import numpy as np
import librosa
import librosa.display
from scipy import signal
import math
import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Flask imports
from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS

# Additional imports for enhanced text processing
try:
    import num2words
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False
    logging.warning("num2words not available - using basic number processing")

try:
    import markdown
    from markdown.extensions import codehilite, fenced_code, tables
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    logging.warning("markdown not available - using regex-based processing")

# --- Configuration ----------------------------------------------------------

KOKORO_VOICES = {
    'af_heart': {'lang': 'en-US', 'gender': 'female', 'description': 'Heart ❤️'},
    'af_bella': {'lang': 'en-US', 'gender': 'female', 'description': 'Bella 🔥'},
    'af_nicole': {'lang': 'en-US', 'gender': 'female', 'description': 'Nicole 🎧'},
    'af_aoede': {'lang': 'en-US', 'gender': 'female', 'description': 'Aoede'},
    'af_kore': {'lang': 'en-US', 'gender': 'female', 'description': 'Kore'},
    'af_sarah': {'lang': 'en-US', 'gender': 'female', 'description': 'Sarah'},
    'af_nova': {'lang': 'en-US', 'gender': 'female', 'description': 'Nova'},
    'af_sky': {'lang': 'en-US', 'gender': 'female', 'description': 'Sky'},
    'af_alloy': {'lang': 'en-US', 'gender': 'female', 'description': 'Alloy'},
    'af_jessica': {'lang': 'en-US', 'gender': 'female', 'description': 'Jessica'},
    'af_river': {'lang': 'en-US', 'gender': 'female', 'description': 'River'},
    'am_michael': {'lang': 'en-US', 'gender': 'male', 'description': 'Michael'},
    'am_fenrir': {'lang': 'en-US', 'gender': 'male', 'description': 'Fenrir'},
    'am_puck': {'lang': 'en-US', 'gender': 'male', 'description': 'Puck'},
    'am_echo': {'lang': 'en-US', 'gender': 'male', 'description': 'Echo'},
    'am_eric': {'lang': 'en-US', 'gender': 'male', 'description': 'Eric'},
    'am_liam': {'lang': 'en-US', 'gender': 'male', 'description': 'Liam'},
    'am_onyx': {'lang': 'en-US', 'gender': 'male', 'description': 'Onyx'},
    'am_santa': {'lang': 'en-US', 'gender': 'male', 'description': 'Santa'},
    'am_adam': {'lang': 'en-US', 'gender': 'male', 'description': 'Adam'},
    'bf_emma': {'lang': 'en-GB', 'gender': 'female', 'description': 'Emma'},
    'bf_isabella': {'lang': 'en-GB', 'gender': 'female', 'description': 'Isabella'},
    'bf_alice': {'lang': 'en-GB', 'gender': 'female', 'description': 'Alice'},
    'bf_lily': {'lang': 'en-GB', 'gender': 'female', 'description': 'Lily'},
    'bm_george': {'lang': 'en-GB', 'gender': 'male', 'description': 'George'},
    'bm_fable': {'lang': 'en-GB', 'gender': 'male', 'description': 'Fable'},
    'bm_lewis': {'lang': 'en-GB', 'gender': 'male', 'description': 'Lewis'},
    'bm_daniel': {'lang': 'en-GB', 'gender': 'male', 'description': 'Daniel'},
}

DEFAULT_VOICE = 'af_heart'
SAMPLE_RATE = 24000
API_PORT = 5000
API_HOST = '0.0.0.0'

ENABLE_CORS = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
LOG_AUTH_ATTEMPTS = True

# Enhanced configuration
MAX_TEXT_LENGTH = 10000
DEFAULT_CHUNK_SIZE = 400
MIN_CHUNK_SIZE = 50
MAX_CHUNKS = 50
CACHE_SIZE = 1000

# --- Setup logging ----------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GPU setup --------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()
logger.info(f"CUDA Available: {CUDA_AVAILABLE}")

# --- Audio Playback Control -------------------------------------------------

class AudioPlaybackController:
    """Global controller for managing audio playback state and interruptions."""
    
    def __init__(self):
        self.is_playing = False
        self.should_stop = False
        self.current_session_id = None
        self.playback_start_time = None
        self.interrupted_at = None
        self.total_duration = None
        self.lock = threading.Lock()
        
    def start_playback(self, session_id: str, duration: float):
        """Start a new playback session."""
        with self.lock:
            self.is_playing = True
            self.should_stop = False
            self.current_session_id = session_id
            self.playback_start_time = time.time()
            self.interrupted_at = None
            self.total_duration = duration
            logger.info(f"Started playback session: {session_id}, duration: {duration:.2f}s")
    
    def stop_playback(self, session_id: str = None):
        """Stop the current playback session."""
        with self.lock:
            if session_id and session_id != self.current_session_id:
                return False  # Wrong session
            
            if self.is_playing:
                current_time = time.time()
                self.interrupted_at = current_time - self.playback_start_time if self.playback_start_time else 0
                logger.info(f"Stopped playback session: {self.current_session_id}, interrupted at: {self.interrupted_at:.2f}s")
            
            self.is_playing = False
            self.should_stop = True
            return True
    
    def finish_playback(self):
        """Mark playback as naturally finished."""
        with self.lock:
            self.is_playing = False
            self.should_stop = False
            self.interrupted_at = None
            logger.info(f"Finished playback session: {self.current_session_id}")
    
    def get_status(self):
        """Get current playback status."""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.playback_start_time if self.playback_start_time else 0
            
            return {
                "is_playing": self.is_playing,
                "session_id": self.current_session_id,
                "elapsed_time": elapsed,
                "total_duration": self.total_duration,
                "interrupted_at": self.interrupted_at,
                "should_stop": self.should_stop
            }

# Global playback controller
playback_controller = AudioPlaybackController()

# --- Enhanced Text Processing Classes ---------------------------------------

@dataclass
class TextChunk:
    """Represents a processed text chunk with metadata."""
    text: str
    original_text: str
    chunk_id: int
    total_chunks: int
    processing_time: float = 0.0
    char_count: int = 0
    
    def __post_init__(self):
        self.char_count = len(self.text)

class TextProcessingMode(Enum):
    """Text processing modes for different input types."""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    SSML = "ssml"

class ProductionTextProcessor:
    """Production-grade text processor with comprehensive normalization."""
    
    def __init__(self):
        # Enhanced character replacements (comprehensive Unicode mapping)
        self.char_replacements = {
            # Smart quotes and apostrophes
            '"': '"', '"': '"', ''': "'", ''': "'",
            '‚': ',', '„': '"', '‹': '<', '›': '>',
            '«': '"', '»': '"',
            
            # Dashes and hyphens
            '–': '-', '—': '-', '―': '-', '‒': '-',
            
            # Mathematical and special symbols
            '×': ' times ', '÷': ' divided by ', '±': ' plus or minus ',
            '≤': ' less than or equal to ', '≥': ' greater than or equal to ',
            '≠': ' not equal to ', '≈': ' approximately ',
            '∞': ' infinity ', '√': ' square root of ',
            
            # Currency symbols
            '€': ' euros ', '£': ' pounds ', '¥': ' yen ',
            '₹': ' rupees ', '₽': ' rubles ', '₩': ' won ',
            
            # Other symbols
            '©': ' copyright ', '®': ' registered ', '™': ' trademark ',
            '§': ' section ', '¶': ' paragraph ', '†': ' dagger ',
            '‡': ' double dagger ', '•': ' bullet ', '‰': ' per mille ',
            '…': '...', '⋯': '...', '⋮': '...',
            
            # Fractions
            '½': ' one half ', '⅓': ' one third ', '⅔': ' two thirds ',
            '¼': ' one quarter ', '¾': ' three quarters ', '⅕': ' one fifth ',
            '⅖': ' two fifths ', '⅗': ' three fifths ', '⅘': ' four fifths ',
            '⅙': ' one sixth ', '⅚': ' five sixths ', '⅛': ' one eighth ',
            '⅜': ' three eighths ', '⅝': ' five eighths ', '⅞': ' seven eighths ',
        }
        
        # Context-aware abbreviations with disambiguation
        self.abbreviations = {
            # Titles
            'Dr.': {'default': 'Doctor', 'context': {'street': 'Drive'}},
            'Mr.': {'default': 'Mister'},
            'Mrs.': {'default': 'Missus'},
            'Ms.': {'default': 'Miss'},
            'Prof.': {'default': 'Professor'},
            
            # Places and directions
            'St.': {'default': 'Saint', 'context': {'address': 'Street'}},
            'Ave.': {'default': 'Avenue'},
            'Blvd.': {'default': 'Boulevard'},
            'Rd.': {'default': 'Road'},
            'Ln.': {'default': 'Lane'},
            'Ct.': {'default': 'Court'},
            'Pl.': {'default': 'Place'},
            'Sq.': {'default': 'Square'},
            'N.': {'default': 'North', 'context': {'name': 'N'}},
            'S.': {'default': 'South', 'context': {'name': 'S'}},
            'E.': {'default': 'East', 'context': {'name': 'E'}},
            'W.': {'default': 'West', 'context': {'name': 'W'}},
            
            # Common abbreviations
            'etc.': {'default': 'etcetera'},
            'vs.': {'default': 'versus'},
            'e.g.': {'default': 'for example'},
            'i.e.': {'default': 'that is'},
            'cf.': {'default': 'compare'},
            'et al.': {'default': 'and others'},
            'ibid.': {'default': 'in the same place'},
            'op. cit.': {'default': 'in the work cited'},
            
            # Business
            'Inc.': {'default': 'Incorporated'},
            'Corp.': {'default': 'Corporation'},
            'Ltd.': {'default': 'Limited'},
            'Co.': {'default': 'Company'},
            'LLC': {'default': 'Limited Liability Company'},
            'LLP': {'default': 'Limited Liability Partnership'},
            
            # Time and dates
            'Jan.': {'default': 'January'},
            'Feb.': {'default': 'February'},
            'Mar.': {'default': 'March'},
            'Apr.': {'default': 'April'},
            'Jun.': {'default': 'June'},
            'Jul.': {'default': 'July'},
            'Aug.': {'default': 'August'},
            'Sep.': {'default': 'September'},
            'Sept.': {'default': 'September'},
            'Oct.': {'default': 'October'},
            'Nov.': {'default': 'November'},
            'Dec.': {'default': 'December'},
            
            'Mon.': {'default': 'Monday'},
            'Tue.': {'default': 'Tuesday'},
            'Wed.': {'default': 'Wednesday'},
            'Thu.': {'default': 'Thursday'},
            'Fri.': {'default': 'Friday'},
            'Sat.': {'default': 'Saturday'},
            'Sun.': {'default': 'Sunday'},
            
            'AM': {'default': 'A M'},
            'PM': {'default': 'P M'},
            'a.m.': {'default': 'A M'},
            'p.m.': {'default': 'P M'},
        }
        
        # Enhanced markdown patterns
        self.markdown_patterns = [
            # Code blocks (must come first)
            (r'```[\s\S]*?```', ' [code block] '),
            (r'`([^`]+)`', r'\1'),
            
            # Headers
            (r'^#{1,6}\s+(.+)$', r'\1', re.MULTILINE),
            
            # Links and images
            (r'!\[([^\]]*)\]$$[^)]+$$', r'\1'),  # Images - use alt text
            (r'\[([^\]]+)\]$$[^)]+$$', r'\1'),   # Links - use link text
            
            # Emphasis
            (r'\*\*\*(.+?)\*\*\*', r'\1'),      # Bold italic
            (r'\*\*(.+?)\*\*', r'\1'),          # Bold
            (r'\*(.+?)\*', r'\1'),              # Italic
            (r'__(.+?)__', r'\1'),              # Bold alt
            (r'_(.+?)_', r'\1'),                # Italic alt
            (r'~~(.+?)~~', r'\1'),              # Strikethrough
            
            # Lists
            (r'^\s*[-*+]\s+(.+)$', r'\1', re.MULTILINE),  # Unordered lists
            (r'^\s*\d+\.\s+(.+)$', r'\1', re.MULTILINE),  # Ordered lists
            
            # Blockquotes
            (r'^\s*>\s*(.+)$', r'\1', re.MULTILINE),
            
            # Horizontal rules
            (r'^[-*_]{3,}$', '', re.MULTILINE),
            
            # Tables (remove pipe separators)
            (r'\|', ' '),
        ]
        
        # Number processing patterns
        self.number_patterns = [
            # Currency with amounts
            (r'(\$|USD|€|EUR|£|GBP|¥|JPY|₹|INR)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', self._expand_currency),
            
            # Percentages
            (r'(\d+(?:\.\d+)?)\s*%', r'\1 percent'),
            
            # Temperatures
            (r'(\d+(?:\.\d+)?)\s*°([CF])', self._expand_temperature),
            
            # Measurements
            (r'(\d+(?:\.\d+)?)\s*(km|m|cm|mm|ft|in|mi|kg|g|lb|oz)', self._expand_measurement),
            
            # Years (4 digits)
            (r'\b(19|20)\d{2}\b', self._expand_year),
            
            # Large numbers with commas
            (r'\b(\d{1,3}(?:,\d{3})+)\b', self._expand_large_number),
            
            # Decimals
            (r'\b(\d+)\.(\d+)\b', self._expand_decimal),
            
            # Ordinals
            (r'\b(\d+)(st|nd|rd|th)\b', self._expand_ordinal),
            
            # Phone numbers (basic pattern)
            (r'\b(\d{3})-(\d{3})-(\d{4})\b', r'\1 \2 \3'),
            
            # Time
            (r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b', self._expand_time),
        ]
        
        # Sentence boundary patterns for smart chunking
        self.sentence_boundaries = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|'  # Period/exclamation/question + space + capital
            r'(?<=[.!?])\s*\n\s*(?=[A-Z])|'  # Same with newline
            r'(?<=\.)\s+(?=["\'"]?[A-Z])'  # Period + space + optional quote + capital
        )
        
        # Compile regex patterns for performance
        self._compile_patterns()
        
        # Cache for processed text
        self._cache = {}
        self._cache_lock = threading.Lock()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.compiled_markdown = []
        for pattern in self.markdown_patterns:
            if len(pattern) == 3:
                self.compiled_markdown.append((re.compile(pattern[0], pattern[2]), pattern[1]))
            else:
                self.compiled_markdown.append((re.compile(pattern[0]), pattern[1]))
    
    def _expand_currency(self, match):
        """Expand currency amounts."""
        symbol = match.group(1)
        amount = match.group(2)
        
        # Currency symbol mapping
        currency_map = {
            '$': 'dollars', 'USD': 'dollars',
            '€': 'euros', 'EUR': 'euros',
            '£': 'pounds', 'GBP': 'pounds',
            '¥': 'yen', 'JPY': 'yen',
            '₹': 'rupees', 'INR': 'rupees'
        }
        
        currency_name = currency_map.get(symbol, 'units')
        
        if HAS_NUM2WORDS:
            try:
                # Remove commas and convert to float
                amount_float = float(amount.replace(',', ''))
                if amount_float == int(amount_float):
                    # Whole number
                    amount_words = num2words.num2words(int(amount_float))
                else:
                    # Has decimal places
                    dollars = int(amount_float)
                    cents = int((amount_float - dollars) * 100)
                    amount_words = f"{num2words.num2words(dollars)} {currency_name}"
                    if cents > 0:
                        amount_words += f" and {num2words.num2words(cents)} cents"
                    return amount_words
                return f"{amount_words} {currency_name}"
            except:
                pass
        
        return f"{amount} {currency_name}"
    
    def _expand_temperature(self, match):
        """Expand temperature readings."""
        temp = match.group(1)
        scale = match.group(2)
        scale_name = 'Celsius' if scale.upper() == 'C' else 'Fahrenheit'
        return f"{temp} degrees {scale_name}"
    
    def _expand_measurement(self, match):
        """Expand measurements."""
        value = match.group(1)
        unit = match.group(2)
        
        unit_map = {
            'km': 'kilometers', 'm': 'meters', 'cm': 'centimeters', 'mm': 'millimeters',
            'ft': 'feet', 'in': 'inches', 'mi': 'miles',
            'kg': 'kilograms', 'g': 'grams', 'lb': 'pounds', 'oz': 'ounces'
        }
        
        unit_name = unit_map.get(unit, unit)
        return f"{value} {unit_name}"
    
    def _expand_year(self, match):
        """Expand years for better pronunciation."""
        year = match.group(0)
        if HAS_NUM2WORDS:
            try:
                return num2words.num2words(int(year))
            except:
                pass
        return year
    
    def _expand_large_number(self, match):
        """Expand large numbers with commas."""
        number = match.group(1).replace(',', '')
        if HAS_NUM2WORDS:
            try:
                return num2words.num2words(int(number))
            except:
                pass
        return number
    
    def _expand_decimal(self, match):
        """Expand decimal numbers."""
        whole = match.group(1)
        decimal = match.group(2)
        
        if HAS_NUM2WORDS:
            try:
                whole_words = num2words.num2words(int(whole))
                decimal_words = ' '.join([num2words.num2words(int(d)) for d in decimal])
                return f"{whole_words} point {decimal_words}"
            except:
                pass
        
        return f"{whole} point {' '.join(decimal)}"
    
    def _expand_ordinal(self, match):
        """Expand ordinal numbers."""
        number = match.group(1)
        suffix = match.group(2)
        
        if HAS_NUM2WORDS:
            try:
                return num2words.num2words(int(number), ordinal=True)
            except:
                pass
        
        return f"{number}{suffix}"
    
    def _expand_time(self, match):
        """Expand time expressions."""
        hour = int(match.group(1))
        minute = match.group(2)
        period = match.group(3) if match.group(3) else ""
        
        if HAS_NUM2WORDS:
            try:
                hour_words = num2words.num2words(hour)
                if minute == "00":
                    time_words = f"{hour_words} o'clock"
                else:
                    minute_words = num2words.num2words(int(minute))
                    time_words = f"{hour_words} {minute_words}"
                
                if period:
                    time_words += f" {period.upper()}"
                
                return time_words
            except:
                pass
        
        return f"{hour} {minute} {period}".strip()
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters while preserving important accents."""
        # First, unescape HTML entities
        text = html.unescape(text)
        
        # Replace special characters
        for char, replacement in self.char_replacements.items():
            text = text.replace(char, replacement)
        
        # Normalize Unicode but preserve accented characters in names
        # This is a balance between ASCII conversion and preserving pronunciation
        normalized = unicodedata.normalize('NFC', text)
        
        return normalized
    
    def clean_markdown(self, text: str) -> str:
        """Clean markdown formatting using proper parsing when available."""
        if HAS_MARKDOWN:
            try:
                # Convert markdown to HTML, then extract text
                md = markdown.Markdown(extensions=['fenced_code', 'tables', 'codehilite'])
                html_content = md.convert(text)
                
                # Simple HTML tag removal (more robust than regex for basic cases)
                import re
                clean_text = re.sub(r'<[^>]+>', '', html_content)
                clean_text = html.unescape(clean_text)
                return clean_text
            except Exception as e:
                logger.warning(f"Markdown parsing failed, falling back to regex: {e}")
        
        # Fallback to regex-based cleaning
        cleaned = text
        for pattern, replacement in self.compiled_markdown:
            cleaned = pattern.sub(replacement, cleaned)
        
        return cleaned
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations with context awareness."""
        result = text
        
        for abbrev, expansion_data in self.abbreviations.items():
            if isinstance(expansion_data, dict):
                default_expansion = expansion_data['default']
                # For now, use default expansion
                # TODO: Implement context detection for better disambiguation
                expansion = default_expansion
            else:
                expansion = expansion_data
            
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
        
        return result
    
    def process_numbers(self, text: str) -> str:
        """Process numbers, currency, and measurements."""
        result = text
        
        for pattern, replacement in self.number_patterns:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                result = re.sub(pattern, replacement, result)
        
        return result
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation for better TTS processing."""
        # Multiple punctuation marks
        text = re.sub(r'\.{2,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'[!]{2,}', '!', text)   # Multiple exclamations
        text = re.sub(r'[?]{2,}', '?', text)   # Multiple questions
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        
        # Multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up extra whitespace
        text = text.strip()
        
        return text
    
    def preserve_case_markers(self, text: str) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Identify and preserve important case information."""
        case_preservations = []
        
        # Find acronyms (2+ consecutive uppercase letters)
        for match in re.finditer(r'\b[A-Z]{2,}\b', text):
            case_preservations.append((match.start(), match.end(), match.group()))
        
        # Find proper nouns at sentence beginnings
        for match in re.finditer(r'(?:^|[.!?]\s+)([A-Z][a-z]+)', text):
            start = match.start(1)
            end = match.end(1)
            case_preservations.append((start, end, match.group(1)))
        
        return text, case_preservations
    
    def smart_chunk_text(self, text: str, max_length: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Intelligently chunk text at sentence boundaries."""
        if len(text) <= max_length:
            return [text]
        
        # First, try to split at sentence boundaries
        sentences = self.sentence_boundaries.split(text)
        if not sentences:
            sentences = [text]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed max_length
            if len(current_chunk + sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    # Single sentence is too long, split at commas or other punctuation
                    sub_parts = re.split(r'(?<=[,;:])\s+', sentence)
                    for part in sub_parts:
                        if len(current_chunk + part) > max_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part + " "
                        else:
                            current_chunk += part + " "
            else:
                current_chunk += sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out chunks that are too short
        valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= MIN_CHUNK_SIZE]
        
        return valid_chunks[:MAX_CHUNKS]  # Limit total chunks
    
    def detect_input_mode(self, text: str) -> TextProcessingMode:
        """Detect the input text format."""
        # Check for SSML
        if '<speak>' in text or '<voice' in text or '<prosody' in text:
            return TextProcessingMode.SSML
        
        # Check for HTML
        if '<html>' in text or '<div>' in text or '<p>' in text:
            return TextProcessingMode.HTML
        
        # Check for Markdown
        markdown_indicators = [
            r'^#{1,6}\s',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\[.*?\]$$.*?$$',  # Links
            r'```',  # Code blocks
            r'^\s*[-*+]\s',  # Lists
            r'^\s*\d+\.\s',  # Numbered lists
        ]
        
        for pattern in markdown_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return TextProcessingMode.MARKDOWN
        
        return TextProcessingMode.PLAIN
    
    def process_text(self, text: str, mode: Optional[TextProcessingMode] = None, 
                    max_chunk_length: int = DEFAULT_CHUNK_SIZE) -> List[TextChunk]:
        """Main text processing pipeline."""
        start_time = time.time()
        
        if not text or not text.strip():
            return []
        
        # Check cache first
        cache_key = hashlib.md5(f"{text}_{max_chunk_length}".encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._cache:
                logger.info("Using cached text processing result")
                return self._cache[cache_key]
        
        original_text = text
        logger.info(f"Processing text: {text[:100]}...")
        
        # Auto-detect mode if not provided
        if mode is None:
            mode = self.detect_input_mode(text)
        
        logger.info(f"Detected input mode: {mode.value}")
        
        # Step 1: Normalize Unicode and HTML entities
        processed = self.normalize_unicode(text)
        
        # Step 2: Handle different input formats
        if mode == TextProcessingMode.MARKDOWN:
            processed = self.clean_markdown(processed)
        elif mode == TextProcessingMode.HTML:
            # Basic HTML cleaning
            processed = re.sub(r'<[^>]+>', '', processed)
            processed = html.unescape(processed)
        elif mode == TextProcessingMode.SSML:
            # For SSML, we might want to preserve some tags
            # For now, just clean basic HTML-like tags
            processed = re.sub(r'<(?!speak|voice|prosody|break|emphasis)[^>]+>', '', processed)
        
        # Step 3: Preserve case information
        processed, case_info = self.preserve_case_markers(processed)
        
        # Step 4: Expand abbreviations
        processed = self.expand_abbreviations(processed)
        
        # Step 5: Process numbers and special formats
        processed = self.process_numbers(processed)
        
        # Step 6: Normalize punctuation
        processed = self.normalize_punctuation(processed)
        
        # Step 7: Smart chunking
        text_chunks = self.smart_chunk_text(processed, max_chunk_length)
        
        # Step 8: Create TextChunk objects
        chunks = []
        processing_time = time.time() - start_time
        
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) >= 2:  # Minimum viable chunk size
                chunk = TextChunk(
                    text=chunk_text.strip(),
                    original_text=original_text,
                    chunk_id=i,
                    total_chunks=len(text_chunks),
                    processing_time=processing_time / len(text_chunks)
                )
                chunks.append(chunk)
        
        # Cache the result
        with self._cache_lock:
            if len(self._cache) >= CACHE_SIZE:
                # Simple cache eviction - remove oldest entries
                oldest_keys = list(self._cache.keys())[:CACHE_SIZE // 2]
                for key in oldest_keys:
                    del self._cache[key]
            self._cache[cache_key] = chunks
        
        logger.info(f"Text processed into {len(chunks)} chunks in {processing_time:.2f}s")
        return chunks

# --- Note to Frequency Conversion --------------------------------------------

NOTE_PATTERN = re.compile(r"^([A-G])([#b]?)(\d)$")
NOTE_INDEX = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}

def note_to_freq(note: str) -> float:
    """Convert scientific pitch (e.g., A4) → frequency in Hz."""
    m = NOTE_PATTERN.match(note)
    if not m:
        raise ValueError(f"Invalid note: {note}")
    letter, accidental, octave = m.groups()
    key = letter + accidental
    semitone = NOTE_INDEX[key]
    octave = int(octave)
    midi = semitone + 12 * (octave + 1)
    return 440.0 * 2 ** ((midi - 69) / 12)

def available_notes() -> List[str]:
    """Generate a list of available musical notes."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return ["None"] + [f"{n}{o}" for o in range(2, 7) for n in names]

# --- Audio Processing Functions with ZERO DEFAULTS -------------------------

def estimate_pitch(audio: np.ndarray, sr: int) -> Optional[float]:
    """Estimate the fundamental frequency of audio."""
    try:
        f0 = librosa.yin(y=audio, fmin=65, fmax=1047, sr=sr)
        median_f0 = np.nanmedian(f0)
        return float(median_f0) if np.any(~np.isnan(f0)) and not np.isnan(median_f0) else None
    except Exception as e:
        logger.warning(f"Pitch estimation failed: {e}")
        return None

def apply_volume(audio: np.ndarray, gain: float = 1.0, gain_db: Optional[float] = None) -> np.ndarray:
    """Apply volume adjustment with linear or dB gain."""
    if gain_db is not None:
        gain = 10 ** (gain_db / 20.0)
    gain = np.clip(gain, 0.0, 2.0)
    if abs(gain - 1.0) < 0.01:
        return audio
    logger.info(f"Applying volume gain: {gain:.2f} (linear), {20 * math.log10(max(gain, 1e-10)):.2f}dB")
    try:
        result = audio * gain
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 1e-6:
            result = result / max_abs_result
        return result
    except Exception as e:
        logger.error(f"Volume adjustment failed: {e}")
        return audio

def shift_to_target(audio: np.ndarray, sr: int, target_note: Optional[str], semitone_shift: float, preserve_formants: bool = False) -> np.ndarray:
    """
    FIXED: Shift audio to target note or by semitones with robust librosa compatibility.
    This function now handles different versions of librosa properly.
    """
    actual_semitone_shift = semitone_shift
    
    # Handle target note conversion
    if target_note and target_note.lower() not in ["none", "", "null"]:
        current_pitch = estimate_pitch(audio, sr)
        if current_pitch is None:
            logger.warning("Could not estimate current pitch. Using semitone_shift if provided.")
        else:
            try:
                target_freq = note_to_freq(target_note)
                actual_semitone_shift = 12 * math.log2(target_freq / current_pitch)
                logger.info(f"Pitch shift: Current={current_pitch:.2f}Hz, Target={target_note}({target_freq:.2f}Hz), Semitones={actual_semitone_shift:.2f}")
            except ValueError as e:
                logger.warning(f"Invalid target note '{target_note}': {e}")
                actual_semitone_shift = semitone_shift
    
    # Skip if no significant shift
    if abs(actual_semitone_shift) < 0.01:
        return audio
    
    logger.info(f"Applying pitch shift of {actual_semitone_shift:.2f} semitones, preserve_formants={preserve_formants}")
    
    try:
        # Try with formant preservation first (newer librosa versions)
        if preserve_formants:
            try:
                # Check if librosa supports the res_type parameter (newer versions)
                import inspect
                pitch_shift_params = inspect.signature(librosa.effects.pitch_shift).parameters
                
                if 'res_type' in pitch_shift_params:
                    # Newer librosa with res_type parameter
                    return librosa.effects.pitch_shift(
                        y=audio, 
                        sr=sr, 
                        n_steps=actual_semitone_shift,
                        res_type='kaiser_fast'  # Faster processing
                    )
                else:
                    # Older librosa without res_type
                    return librosa.effects.pitch_shift(
                        y=audio, 
                        sr=sr, 
                        n_steps=actual_semitone_shift
                    )
            except Exception as formant_error:
                logger.warning(f"Formant-preserving pitch shift failed: {formant_error}")
                # Fall back to basic pitch shift
                return librosa.effects.pitch_shift(
                    y=audio, 
                    sr=sr, 
                    n_steps=actual_semitone_shift
                )
        else:
            # Basic pitch shift without formant preservation
            return librosa.effects.pitch_shift(
                y=audio, 
                sr=sr, 
                n_steps=actual_semitone_shift
            )
            
    except Exception as e:
        logger.error(f"Pitch shift failed completely: {e}")
        logger.info("Attempting alternative pitch shift method...")
        
        # Alternative method using phase vocoder if librosa.effects.pitch_shift fails
        try:
            # Use time stretching + resampling as fallback
            stretch_factor = 2 ** (-actual_semitone_shift / 12)
            stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
            
            # Resample back to original length to maintain timing
            target_length = len(audio)
            if len(stretched) != target_length:
                stretched = librosa.util.fix_length(stretched, size=target_length)
            
            return stretched
            
        except Exception as fallback_error:
            logger.error(f"Alternative pitch shift method also failed: {fallback_error}")
            return audio

def apply_formant_shift(audio: np.ndarray, sr: int, shift_factor: float, scale: float = 1.0) -> np.ndarray:
    """Apply formant shifting with intensity scaling."""
    if abs(shift_factor - 1.0) < 0.01:
        return audio
    shift_factor = np.clip(shift_factor, 0.5, 1.5)
    scale = np.clip(scale, 0.5, 2.0)
    effective_shift = 1.0 + (shift_factor - 1.0) * scale
    logger.info(f"Applying formant shift: factor={shift_factor:.2f}, scale={scale:.2f}, effective={effective_shift:.2f}")
    try:
        audio_float32 = audio.astype(np.float32)
        stft_result = librosa.stft(audio_float32)
        magnitude = np.abs(stft_result)
        phase = np.angle(stft_result)
        shifted_magnitude = np.zeros_like(magnitude)
        n_freq_bins = magnitude.shape[0]
        for i in range(magnitude.shape[1]):
            freq_profile = magnitude[:, i]
            source_freq_coords = np.arange(n_freq_bins, dtype=float) / effective_shift
            shifted_profile_frame = np.interp(
                source_freq_coords,
                np.arange(n_freq_bins, dtype=float),
                freq_profile,
                left=freq_profile[0] if len(freq_profile) > 0 else 0.0,
                right=freq_profile[-1] if len(freq_profile) > 0 else 0.0
            )
            shifted_magnitude[:, i] = shifted_profile_frame
        audio_stft_shifted = shifted_magnitude * np.exp(1j * phase)
        audio_shifted = librosa.istft(audio_stft_shifted, length=len(audio_float32))
        max_abs_audio = np.max(np.abs(audio_shifted))
        if max_abs_audio > 1e-6:
            audio_shifted = audio_shifted / max_abs_audio
        return audio_shifted.astype(audio.dtype)
    except Exception as e:
        logger.error(f"Formant shift failed: {e}")
        return audio

def apply_reverb(audio: np.ndarray, sr: int, room_size: float = 0.0, damping: float = 0.5, pre_delay_ms: float = 0.0, stereo_width: float = 0.0) -> np.ndarray:
    """Apply reverb with pre-delay and stereo width. Only applies if room_size > 0."""
    if room_size <= 0.001:
        return audio
    room_size = np.clip(room_size, 0.0, 1.0)
    damping = np.clip(damping, 0.0, 1.0)
    pre_delay_ms = np.clip(pre_delay_ms, 0.0, 100.0)
    stereo_width = np.clip(stereo_width, 0.0, 1.0)
    logger.info(f"Applying reverb: RoomSize={room_size:.2f}, Damping={damping:.2f}, PreDelay={pre_delay_ms:.1f}ms, StereoWidth={stereo_width:.2f}")
    try:
        reverb_length_sec = room_size * 1.5
        pre_delay_samples = int(sr * pre_delay_ms / 1000)
        reverb_length_samples = int(sr * reverb_length_sec) + pre_delay_samples
        if reverb_length_samples <= 0:
            return audio
        time_points = np.arange(reverb_length_samples) / sr
        decay_rate = 5.0 + (1.0 - damping) * 15.0
        decay_envelope = np.exp(-decay_rate * time_points)
        impulse = np.random.randn(reverb_length_samples) * decay_envelope
        impulse_energy_sq = np.sum(impulse**2)
        if impulse_energy_sq > 1e-12:
            impulse = impulse / np.sqrt(impulse_energy_sq)
        else:
            return audio
        if stereo_width > 0:
            impulse_l = impulse * np.sqrt(1 - stereo_width / 2)
            impulse_r = np.random.randn(reverb_length_samples) * decay_envelope * np.sqrt(stereo_width / 2)
            impulse_r_energy = np.sum(impulse_r**2)
            if impulse_r_energy > 1e-12:
                impulse_r = impulse_r / np.sqrt(impulse_r_energy)
            reverb_l = signal.convolve(audio, impulse_l, mode='full')[:len(audio)]
            reverb_r = signal.convolve(audio, impulse_r, mode='full')[:len(audio)]
            reverb_audio = (reverb_l + reverb_r) / 2
        else:
            reverb_audio = signal.convolve(audio, impulse, mode='full')[:len(audio)]
        max_abs_reverb = np.max(np.abs(reverb_audio))
        if max_abs_reverb > 1e-6:
            reverb_audio = reverb_audio / max_abs_reverb
        dry_gain = 0.7
        wet_gain = 0.1 + room_size * 0.4
        result = dry_gain * audio + wet_gain * reverb_audio
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 1e-6:
            result = result / max_abs_result
        return result
    except Exception as e:
        logger.error(f"Reverb failed: {e}")
        return audio

def apply_eq(audio: np.ndarray, sr: int, bands: List[Dict[str, Any]]) -> np.ndarray:
    """Apply parametric equalizer with multiple bands. Only applies if bands are provided."""
    if not bands:
        return audio
    logger.info(f"Applying EQ with {len(bands)} bands")
    try:
        processed_audio = audio.copy()
        nyquist = sr / 2.0
        for band in bands:
            freq = np.clip(band.get('frequency_hz', 1000.0), 20.0, nyquist - 1e-5)
            gain_db = np.clip(band.get('gain_db', 0.0), -24.0, 24.0)
            q_factor = np.clip(band.get('q_factor', 1.0), 0.1, 10.0)
            band_type = band.get('type', 'peak').lower()
            if abs(gain_db) < 0.1:
                continue
            logger.info(f"EQ Band: Type={band_type}, Freq={freq:.1f}Hz, Gain={gain_db:.1f}dB, Q={q_factor:.2f}")
            try:
                if band_type == 'peak':
                    sos = signal.iirpeak(w0=freq, Q=q_factor, gain_db=gain_db, fs=sr)
                    processed_audio = signal.sosfiltfilt(sos, processed_audio)
                elif band_type == 'low_shelf':
                    try:
                        sos = signal.iirshelf(w0=freq, Q=0.707, gain_db=gain_db, fs=sr, ftype='AB')
                        processed_audio = signal.sosfiltfilt(sos, processed_audio)
                    except AttributeError:
                        logger.info(f"Fallback to Butterworth for low_shelf at {freq}Hz")
                        gain_linear = 10 ** (gain_db / 20.0)
                        norm_freq = freq / nyquist
                        sos = signal.butter(2, norm_freq, btype='lowpass', output='sos')
                        low_freq = signal.sosfiltfilt(sos, processed_audio)
                        processed_audio = processed_audio + (gain_linear - 1.0) * low_freq
                elif band_type == 'high_shelf':
                    try:
                        sos = signal.iirshelf(w0=freq, Q=0.707, gain_db=gain_db, fs=sr, ftype='AB')
                        processed_audio = signal.sosfiltfilt(sos, processed_audio)
                    except AttributeError:
                        logger.info(f"Fallback to Butterworth for high_shelf at {freq}Hz")
                        gain_linear = 10 ** (gain_db / 20.0)
                        norm_freq = freq / nyquist
                        sos = signal.butter(2, norm_freq, btype='highpass', output='sos')
                        high_freq = signal.sosfiltfilt(sos, processed_audio)
                        processed_audio = processed_audio + (gain_linear - 1.0) * high_freq
                else:
                    logger.warning(f"Unknown EQ band type: {band_type}")
            except Exception as e:
                logger.warning(f"EQ band failed: {e}")
        max_abs_eq = np.max(np.abs(processed_audio))
        if max_abs_eq > 1e-6:
            processed_audio = processed_audio / max_abs_eq
        return processed_audio
    except Exception as e:
        logger.error(f"EQ failed: {e}")
        return audio

def apply_distortion(audio: np.ndarray, drive_db: float = 0.0, dist_type: str = 'tanh', mix: float = 0.0) -> np.ndarray:
    """Apply distortion effect. Only applies if drive_db > 0 and mix > 0."""
    drive_db = np.clip(drive_db, 0.0, 36.0)
    mix = np.clip(mix, 0.0, 1.0)
    dist_type = dist_type.lower()
    if drive_db < 0.1 or mix < 0.001:
        return audio
    logger.info(f"Applying distortion: Type={dist_type}, Drive={drive_db:.1f}dB, Mix={mix:.2f}")
    try:
        drive_gain = 10 ** (drive_db / 20.0)
        distorted = audio * drive_gain
        if dist_type == 'tanh':
            distorted = np.tanh(distorted * 2.0) * 0.8
        elif dist_type == 'soft':
            distorted = np.clip(distorted, -1.0, 1.0)
        elif dist_type == 'hard':
            distorted = np.sign(distorted) * np.minimum(np.abs(distorted), 1.0)
        else:
            logger.warning(f"Unknown distortion type: {dist_type}, using tanh")
            distorted = np.tanh(distorted * 2.0) * 0.8
        max_abs_dist = np.max(np.abs(distorted))
        if max_abs_dist > 1e-6:
            distorted = distorted / max_abs_dist
        result = (1.0 - mix) * audio + mix * distorted
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 1e-6:
            result = result / max_abs_result
        return result
    except Exception as e:
        logger.error(f"Distortion failed: {e}")
        return audio

def apply_chorus(audio: np.ndarray, sr: int, delay_ms: float = 0.0, depth: float = 0.0, rate_hz: float = 0.0, mix: float = 0.0) -> np.ndarray:
    """Apply chorus effect. Only applies if depth > 0 and mix > 0."""
    delay_ms = np.clip(delay_ms, 5.0, 50.0)
    depth = np.clip(depth, 0.0, 0.1)
    rate_hz = np.clip(rate_hz, 0.1, 5.0)
    mix = np.clip(mix, 0.0, 1.0)
    
    # Early return if effect should not be applied
    if depth < 0.001 or mix < 0.001:
        return audio
        
    logger.info(f"Applying chorus: Delay={delay_ms:.1f}ms, Depth={depth:.3f}, Rate={rate_hz:.2f}Hz, Mix={mix:.2f}")
    try:
        delay_samples = int(sr * delay_ms / 1000)
        if delay_samples <= 0:
            return audio
            
        t = np.arange(len(audio)) / sr
        mod = depth * np.sin(2 * np.pi * rate_hz * t)
        indices = np.arange(len(audio)) - delay_samples * (1.0 + mod)
        indices = np.clip(indices, 0, len(audio) - 1)
        chorus_audio = np.interp(np.arange(len(audio)), indices, audio)
        
        max_abs_chorus = np.max(np.abs(chorus_audio))
        if max_abs_chorus > 1e-6:
            chorus_audio = chorus_audio / max_abs_chorus
            
        result = (1.0 - mix) * audio + mix * chorus_audio
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 1e-6:
            result = result / max_abs_result
        return result
    except Exception as e:
        logger.error(f"Chorus failed: {e}")
        return audio

def apply_flanger(audio: np.ndarray, sr: int, delay_ms: float = 0.0, depth: float = 0.0, rate_hz: float = 0.0, feedback: float = 0.0, mix: float = 0.0) -> np.ndarray:
    """Apply flanger effect. Only applies if depth > 0 and mix > 0."""
    delay_ms = np.clip(delay_ms, 0.1, 10.0)
    depth = np.clip(depth, 0.0, 0.05)
    rate_hz = np.clip(rate_hz, 0.1, 10.0)
    feedback = np.clip(feedback, 0.0, 0.9)
    mix = np.clip(mix, 0.0, 1.0)
    
    # Early return if effect should not be applied
    if depth < 0.001 or mix < 0.001:
        return audio
        
    logger.info(f"Applying flanger: Delay={delay_ms:.1f}ms, Depth={depth:.3f}, Rate={rate_hz:.2f}Hz, Feedback={feedback:.2f}, Mix={mix:.2f}")
    try:
        delay_samples = int(sr * delay_ms / 1000)
        if delay_samples <= 0:
            return audio
            
        t = np.arange(len(audio)) / sr
        mod = depth * np.sin(2 * np.pi * rate_hz * t)
        output = audio.copy()
        delay_buffer = np.zeros(len(audio) + delay_samples)
        delay_buffer[:len(audio)] = audio
        
        for i in range(len(audio)):
            delay_time = delay_samples * (1.0 + mod[i])
            idx = i - delay_time
            if idx >= 0:
                interp_idx = int(idx)
                frac = idx - interp_idx
                if interp_idx + 1 < len(delay_buffer):
                    delayed_sample = (1 - frac) * delay_buffer[interp_idx] + frac * delay_buffer[interp_idx + 1]
                    output[i] += feedback * delayed_sample
                    delay_buffer[i + delay_samples] += feedback * delayed_sample
                    
        max_abs_flanger = np.max(np.abs(output))
        if max_abs_flanger > 1e-6:
            output = output / max_abs_flanger
            
        result = (1.0 - mix) * audio + mix * output
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 1e-6:
            result = result / max_abs_result
        return result
    except Exception as e:
        logger.error(f"Flanger failed: {e}")
        return audio

def apply_compression(audio: np.ndarray, sr: int, threshold_db: float = 0.0, ratio: float = 1.0, attack_ms: float = 0.1, release_ms: float = 10.0) -> np.ndarray:
    """Apply dynamic range compression. Only applies if ratio > 1.0 and threshold_db < 0."""
    threshold_db = np.clip(threshold_db, -60.0, 0.0)
    ratio = np.clip(ratio, 1.0, 20.0)
    attack_ms = np.clip(attack_ms, 0.1, 100.0)
    release_ms = np.clip(release_ms, 10.0, 1000.0)
    
    # Early return if compression should not be applied
    if ratio <= 1.01 or threshold_db >= -0.1:
        return audio
        
    logger.info(f"Applying compression: Threshold={threshold_db:.1f}dB, Ratio={ratio:.1f}, Attack={attack_ms:.1f}ms, Release={release_ms:.1f}ms")
    try:
        threshold = 10 ** (threshold_db / 20.0)
        attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
        envelope = np.zeros_like(audio)
        gain = np.ones_like(audio)
        
        for i in range(len(audio)):
            envelope[i] = abs(audio[i]) if i == 0 else (1 - attack_coeff) * abs(audio[i]) + attack_coeff * envelope[i - 1]
            if envelope[i] > threshold:
                excess = envelope[i] / threshold
                gain_reduction = threshold * (excess ** (1 / ratio - 1))
                target_gain = gain_reduction / envelope[i] if envelope[i] > 1e-6 else 1.0
            else:
                target_gain = 1.0
            gain[i] = (1 - release_coeff) * target_gain + release_coeff * (gain[i - 1] if i > 0 else 1.0)
            
        result = audio * gain
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 1e-6:
            result = result / max_abs_result
        return result
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        return audio

def apply_voice_character(audio: np.ndarray, sr: int, character: str, params: Optional[Dict] = None) -> np.ndarray:
    """Apply voice character transformation. Only applies if character is not 'none'."""
    if character == "none" or not character:
        return audio
    params = params or {}
    logger.info(f"Applying voice character: {character} with params {params}")
    try:
        result = audio.copy()
        if character == "child":
            pitch_shift = params.get('pitch_shift', 3.0)
            speed = params.get('speed', 1.1)
            formant_shift = params.get('formant_shift', 1.2)
            result = shift_to_target(result, sr, None, pitch_shift, False)
            result = librosa.effects.time_stretch(y=result, rate=speed)
            result = apply_formant_shift(result, sr, formant_shift)
        elif character == "robot":
            pitch_shift = params.get('pitch_shift', 0.0)
            if abs(pitch_shift) > 0.01:
                result = shift_to_target(result, sr, None, pitch_shift, False)
            t = np.arange(len(result)) / sr
            carrier = np.sin(2 * np.pi * params.get('carrier_freq', 80.0) * t)
            result = result * carrier
            result = np.tanh(result * params.get('distortion_factor', 2.5)) * 0.8
        elif character == "deep":
            pitch_shift = params.get('pitch_shift', -4.0)
            speed = params.get('speed', 0.9)
            formant_shift = params.get('formant_shift', 0.8)
            result = shift_to_target(result, sr, None, pitch_shift, False)
            result = librosa.effects.time_stretch(y=result, rate=speed)
            result = apply_formant_shift(result, sr, formant_shift)
        elif character == "whisper":
            b, a = signal.butter(4, 3000 / (sr / 2), 'low')
            result = signal.filtfilt(b, a, result)
            noise = np.random.normal(0, params.get('noise_level', 0.03), len(result)).astype(result.dtype)
            result = result * params.get('signal_level', 0.6) + noise
            result = np.tanh(result * params.get('compression_factor', 1.2)) * 0.9
        elif character == "alien":
            pitch_shift = params.get('pitch_shift', 2.0)
            result = shift_to_target(result, sr, None, pitch_shift, False)
            result = apply_flanger(result, sr, delay_ms=params.get('flanger_delay_ms', 5.0), depth=0.02, rate_hz=0.3, feedback=0.6, mix=0.6)
        elif character == "monster":
            pitch_shift = params.get('pitch_shift', -6.0)
            formant_shift = params.get('formant_shift', 0.7)
            result = shift_to_target(result, sr, None, pitch_shift, False)
            result = apply_formant_shift(result, sr, formant_shift)
            result = apply_distortion(result, drive_db=params.get('distortion_drive', 12.0), dist_type='tanh', mix=0.7)
        elif character == "echo":
            result = apply_reverb(result, sr, room_size=0.6, damping=0.3, pre_delay_ms=params.get('pre_delay_ms', 50.0), stereo_width=0.8)
        else:
            logger.warning(f"Unknown voice character: {character}")
            return audio
        max_abs_result = np.max(np.abs(result))
        if max_abs_result > 1e-6:
            result = result / max_abs_result
        return result
    except Exception as e:
        logger.error(f"Voice character transformation failed: {e}")
        return audio

# --- Enhanced Model Management -----------------------------------------------

class EnhancedModelManager:
    """Production-grade model manager with robust text processing and caching."""
    
    def __init__(self):
        self.models: Dict[bool, KModel] = {
            False: KModel().to('cpu').eval()
        }
        if CUDA_AVAILABLE:
            self.models[True] = KModel().to('cuda').eval()
        
        self.pipelines: Dict[str, KPipeline] = {
            'a': KPipeline(lang_code='a'),
            'b': KPipeline(lang_code='b')
        }
        
        # Enhanced lexicon
        self.pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
        self.pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
        
        # Initialize text processor
        self.text_processor = ProductionTextProcessor()
        
        # Audio cache for processed chunks
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Pre-loading all voices...")
        for voice_name in KOKORO_VOICES.keys():
            lang_code = voice_name[0]
            if lang_code in self.pipelines:
                self.pipelines[lang_code].load_voice(voice_name)
            else:
                logger.error(f"Unknown lang code '{lang_code}' for voice '{voice_name}'")
        logger.info("All voices loaded.")
    
    def generate_audio_for_chunk(self, chunk: TextChunk, voice_name: str, speed: float = 1.0, use_gpu: bool = True) -> torch.Tensor:
        """Generate audio for a single text chunk."""
        # Create cache key
        cache_key = hashlib.md5(f"{chunk.text}_{voice_name}_{speed}".encode()).hexdigest()
        
        with self.cache_lock:
            if cache_key in self.audio_cache:
                logger.info(f"Using cached audio for chunk {chunk.chunk_id}")
                return self.audio_cache[cache_key]
        
        if not chunk.text or len(chunk.text.strip()) < 2:
            raise ValueError(f"Chunk {chunk.chunk_id} text is too short or empty")
        
        lang_code = voice_name[0]
        if lang_code not in self.pipelines:
            raise ValueError(f"No pipeline for lang code '{lang_code}'")
        
        pipeline = self.pipelines[lang_code]
        pack = pipeline.load_voice(voice_name)
        effective_use_gpu = use_gpu and CUDA_AVAILABLE
        
        for _, ps, _ in pipeline(chunk.text, voice_name, speed):
            ref_s = pack[len(ps) - 1]
            try:
                logger.info(f"Generating chunk {chunk.chunk_id}/{chunk.total_chunks} with {'GPU' if effective_use_gpu else 'CPU'} for voice {voice_name}")
                audio_tensor = self.models[effective_use_gpu](ps, ref_s, speed)
                
                # Cache the result
                with self.cache_lock:
                    if len(self.audio_cache) >= CACHE_SIZE:
                        # Simple cache eviction
                        oldest_keys = list(self.audio_cache.keys())[:CACHE_SIZE // 2]
                        for key in oldest_keys:
                            del self.audio_cache[key]
                    self.audio_cache[cache_key] = audio_tensor
                
                return audio_tensor
            except Exception as e:
                logger.error(f"Error on {'GPU' if effective_use_gpu else 'CPU'} for chunk {chunk.chunk_id}: {e}")
                if effective_use_gpu:
                    logger.info("Retrying on CPU...")
                    audio_tensor = self.models[False](ps, ref_s, speed)
                    
                    with self.cache_lock:
                        if len(self.audio_cache) >= CACHE_SIZE:
                            oldest_keys = list(self.audio_cache.keys())[:CACHE_SIZE // 2]
                            for key in oldest_keys:
                                del self.audio_cache[key]
                        self.audio_cache[cache_key] = audio_tensor
                    
                    return audio_tensor
                else:
                    raise
        
        raise RuntimeError(f"Kokoro TTS pipeline yielded no audio frames for chunk {chunk.chunk_id}")
    
    def generate_audio_robust(self, text: str, voice_name: str, speed: float = 1.0, 
                            use_gpu: bool = True, max_chunk_length: int = DEFAULT_CHUNK_SIZE,
                            processing_mode: Optional[TextProcessingMode] = None) -> List[torch.Tensor]:
        """
        Generate audio with robust text processing and chunking support.
        
        Returns list of audio tensors for each processed chunk.
        """
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long. Maximum length is {MAX_TEXT_LENGTH} characters.")
        
        # Process text into manageable chunks
        text_chunks = self.text_processor.process_text(
            text, 
            mode=processing_mode, 
            max_chunk_length=max_chunk_length
        )
        
        if not text_chunks:
            raise ValueError("No valid text chunks after processing")
        
        logger.info(f"Processing {len(text_chunks)} chunks for voice {voice_name}")
        
        audio_tensors = []
        failed_chunks = []
        
        # Process chunks sequentially for now (could be parallelized)
        for chunk in text_chunks:
            try:
                audio_tensor = self.generate_audio_for_chunk(chunk, voice_name, speed, use_gpu)
                audio_tensors.append(audio_tensor)
                logger.info(f"Successfully generated audio for chunk {chunk.chunk_id + 1}/{len(text_chunks)}")
            except Exception as e:
                logger.error(f"Failed to generate audio for chunk {chunk.chunk_id + 1}: {e}")
                failed_chunks.append(chunk.chunk_id)
                continue
        
        if not audio_tensors:
            raise RuntimeError("Failed to generate audio for any text chunks")
        
        if failed_chunks:
            logger.warning(f"Failed to process chunks: {failed_chunks}")
        
        return audio_tensors
    
    def concatenate_audio_tensors(self, audio_tensors: List[torch.Tensor], silence_duration: float = 0.2) -> torch.Tensor:
        """Concatenate multiple audio tensors with configurable silence between them."""
        if len(audio_tensors) == 1:
            return audio_tensors[0]
        
        # Adaptive silence based on content
        if len(audio_tensors) > 10:
            silence_duration = min(silence_duration, 0.15)  # Shorter pauses for many chunks
        
        # Create silence tensor
        silence_samples = int(SAMPLE_RATE * silence_duration)
        silence = torch.zeros(silence_samples, dtype=audio_tensors[0].dtype, device=audio_tensors[0].device)
        
        # Concatenate with silence
        result_parts = []
        for i, tensor in enumerate(audio_tensors):
            result_parts.append(tensor)
            if i < len(audio_tensors) - 1:  # Don't add silence after last chunk
                result_parts.append(silence)
        
        return torch.cat(result_parts, dim=0)
    
    def generate_audio(self, text: str, voice_name: str, speed: float = 1.0, use_gpu: bool = True) -> torch.Tensor:
        """Legacy method for backward compatibility."""
        if not text or len(text.strip()) < 2:
            raise ValueError("Input text is too short or empty")
        
        lang_code = voice_name[0]
        if lang_code not in self.pipelines:
            raise ValueError(f"No pipeline for lang code '{lang_code}'")
        
        pipeline = self.pipelines[lang_code]
        pack = pipeline.load_voice(voice_name)
        effective_use_gpu = use_gpu and CUDA_AVAILABLE
        
        for _, ps, _ in pipeline(text, voice_name, speed):
            ref_s = pack[len(ps) - 1]
            try:
                logger.info(f"Generating with {'GPU' if effective_use_gpu else 'CPU'} for voice {voice_name}")
                audio_tensor = self.models[effective_use_gpu](ps, ref_s, speed)
                return audio_tensor
            except Exception as e:
                logger.error(f"Error on {'GPU' if effective_use_gpu else 'CPU'}: {e}")
                if effective_use_gpu:
                    logger.info("Retrying on CPU...")
                    audio_tensor = self.models[False](ps, ref_s, speed)
                    return audio_tensor
                else:
                    raise
        
        raise RuntimeError("Kokoro TTS pipeline yielded no audio frames")

# Initialize singleton model manager
enhanced_model_manager = EnhancedModelManager()

# --- Flask App Setup --------------------------------------------------------

app = Flask(__name__)
if ENABLE_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for read-aloud compatibility
    logger.info("CORS enabled for all routes")
else:
    logger.info("CORS disabled")

# --- Authentication Middleware ----------------------------------------------

def check_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if LOG_AUTH_ATTEMPTS:
            auth_header = request.headers.get('Authorization')
            if auth_header:
                logger.info(f"Auth header: {auth_header[:30]}...")
            else:
                logger.info("No authentication provided")
        return f(*args, **kwargs)
    return decorated

# --- Helper Functions -------------------------------------------------------

def get_kokoro_voice(voice_id: str) -> str:
    if voice_id in KOKORO_VOICES:
        return voice_id
    logger.warning(f"Unknown voice '{voice_id}', using default '{DEFAULT_VOICE}'")
    return DEFAULT_VOICE

def tensor_to_numpy(audio_tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array."""
    audio_numpy = audio_tensor.cpu().numpy().squeeze()
    if audio_numpy.dtype != np.float32:
        audio_numpy = audio_numpy.astype(np.float32)
    max_val = np.max(np.abs(audio_numpy))
    if max_val > 1.0:
        audio_numpy /= max_val
    return audio_numpy

def numpy_to_format(audio_numpy: np.ndarray, sr: int, audio_format: str) -> bytes:
    """Convert NumPy audio array to specified format."""
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    if audio_format == 'wav':
        buffer = io.BytesIO()
        wavfile.write(buffer, sr, audio_int16)
        return buffer.getvalue()
    elif audio_format == 'mp3':
        audio_segment = AudioSegment(
            data=audio_int16.tobytes(),
            sample_width=2,
            frame_rate=sr,
            channels=1
        )
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="mp3", bitrate="192k")
        return buffer.getvalue()
    else:
        raise ValueError(f"Unsupported audio format: {audio_format}")

def calculate_audio_duration(audio_numpy: np.ndarray, sr: int) -> float:
    """Calculate audio duration in seconds."""
    return len(audio_numpy) / sr

def play_audio_windows_with_interrupt(audio_numpy: np.ndarray, sr: int, session_id: str):
    """Play audio on Windows with interrupt capability."""
    try:
        import winsound
        
        # Calculate duration and start playback session
        duration = calculate_audio_duration(audio_numpy, sr)
        playback_controller.start_playback(session_id, duration)
        
        # Convert to WAV bytes
        wav_bytes = numpy_to_format(audio_numpy, sr, 'wav')
        
        # Check for interruption before playing
        if playback_controller.should_stop:
            playback_controller.finish_playback()
            return
        
        # Play audio (this is blocking)
        winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
        
        # Mark as finished if not interrupted
        if not playback_controller.should_stop:
            playback_controller.finish_playback()
            logger.info("Audio playback completed successfully")
        else:
            logger.info("Audio playback was interrupted")
            
    except ImportError:
        logger.error("winsound module not found")
        playback_controller.finish_playback()
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        playback_controller.finish_playback()

def apply_effects_pipeline(audio_numpy: np.ndarray, effects_params: Dict, effect_order: List[str]) -> np.ndarray:
    """Apply effects pipeline to audio with proper zero-default handling."""
    for effect in effect_order:
        if effect == 'volume':
            volume_params = effects_params.get('volume', {})
            gain = volume_params.get('gain', 1.0)
            gain_db = volume_params.get('gain_db')
            # Only apply if gain != 1.0 or gain_db is specified
            if gain != 1.0 or gain_db is not None:
                audio_numpy = apply_volume(audio_numpy, gain, gain_db)
                
        elif effect == 'equalizer':
            eq_params = effects_params.get('equalizer', {})
            bands = eq_params.get('bands', [])
            # Only apply if bands are provided
            if bands:
                audio_numpy = apply_eq(audio_numpy, SAMPLE_RATE, bands)
                
        elif effect == 'compression':
            comp_params = effects_params.get('compression', {})
            threshold_db = comp_params.get('threshold_db', 0.0)
            ratio = comp_params.get('ratio', 1.0)
            attack_ms = comp_params.get('attack_ms', 0.1)
            release_ms = comp_params.get('release_ms', 10.0)
            # Only apply if compression parameters indicate it should be used
            if threshold_db < 0.0 and ratio > 1.0:
                audio_numpy = apply_compression(audio_numpy, SAMPLE_RATE, threshold_db, ratio, attack_ms, release_ms)
                
        elif effect == 'distortion':
            dist_params = effects_params.get('distortion', {})
            drive_db = dist_params.get('drive_db', 0.0)
            dist_type = dist_params.get('type', 'tanh')
            mix = dist_params.get('mix', 0.0)
            # Only apply if drive_db > 0 and mix > 0
            if drive_db > 0.0 and mix > 0.0:
                audio_numpy = apply_distortion(audio_numpy, drive_db, dist_type, mix)
                
        elif effect == 'pitch':
            pitch_params = effects_params.get('pitch', {})
            target_note = pitch_params.get('target_note')
            semitone_shift = float(pitch_params.get('semitone_shift', 0.0))
            preserve_formants = pitch_params.get('preserve_formants', False)
            # Only apply if target_note is specified or semitone_shift != 0
            if target_note or abs(semitone_shift) > 0.01:
                audio_numpy = shift_to_target(audio_numpy, SAMPLE_RATE, target_note, semitone_shift, preserve_formants)
                
        elif effect == 'formant':
            formant_params = effects_params.get('formant', {})
            shift_percent = float(formant_params.get('shift_percent', 0.0))
            scale = float(formant_params.get('scale', 1.0))
            # Only apply if shift_percent != 0
            if abs(shift_percent) > 0.01:
                shift_factor = 1.0 + (np.clip(shift_percent, -100.0, 100.0) / 200.0)
                audio_numpy = apply_formant_shift(audio_numpy, SAMPLE_RATE, shift_factor, scale)
                
        elif effect == 'voice_character':
            char_params = effects_params.get('voice_character', {})
            char_type = char_params.get('type', 'none')
            char_custom_params = char_params.get('params', {})
            # Only apply if character type is not 'none'
            if char_type != 'none':
                audio_numpy = apply_voice_character(audio_numpy, SAMPLE_RATE, char_type, char_custom_params)
                
        elif effect == 'chorus':
            chorus_params = effects_params.get('chorus', {})
            delay_ms = chorus_params.get('delay_ms', 0.0)
            depth = chorus_params.get('depth', 0.0)
            rate_hz = chorus_params.get('rate_hz', 0.0)
            mix = chorus_params.get('mix', 0.0)
            # Only apply if depth > 0 and mix > 0
            if depth > 0.0 and mix > 0.0:
                audio_numpy = apply_chorus(audio_numpy, SAMPLE_RATE, delay_ms, depth, rate_hz, mix)
                
        elif effect == 'flanger':
            flanger_params = effects_params.get('flanger', {})
            delay_ms = flanger_params.get('delay_ms', 0.0)
            depth = flanger_params.get('depth', 0.0)
            rate_hz = flanger_params.get('rate_hz', 0.0)
            feedback = flanger_params.get('feedback', 0.0)
            mix = flanger_params.get('mix', 0.0)
            # Only apply if depth > 0 and mix > 0
            if depth > 0.0 and mix > 0.0:
                audio_numpy = apply_flanger(audio_numpy, SAMPLE_RATE, delay_ms, depth, rate_hz, feedback, mix)
                
        elif effect == 'reverb':
            reverb_params = effects_params.get('reverb', {})
            room_size = float(reverb_params.get('room_size_percent', 0.0)) / 100.0
            damping = float(reverb_params.get('damping_percent', 50.0)) / 100.0
            pre_delay_ms = float(reverb_params.get('pre_delay_ms', 0.0))
            stereo_width = float(reverb_params.get('stereo_width', 0.0))
            # Only apply if room_size > 0
            if room_size > 0.01:
                audio_numpy = apply_reverb(audio_numpy, SAMPLE_RATE, room_size, damping, pre_delay_ms, stereo_width)
        else:
            logger.warning(f"Unknown effect in order: {effect}")
    
    return audio_numpy

# --- API Endpoints ----------------------------------------------------------

@app.route('/ping', methods=['GET'])
def ping_route():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat() + 'Z'})

@app.route('/v1/audio/speech', methods=['POST'])
@check_auth
def create_speech_route():
    """Standard speech generation endpoint with clean zero-default effects."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('input', '').strip()
        if not text:
            return jsonify({"error": "Missing or empty 'input' field"}), 400
        
        voice_id = data.get('voice', DEFAULT_VOICE)
        speed = float(data.get('speed', 1.0))
        response_format = data.get('response_format', 'mp3').lower()
        use_gpu_flag = data.get('use_gpu', True)
        
        # Default effect order - but effects won't apply unless explicitly configured
        effect_order = data.get('effects', {}).get('order', [
            'volume', 'equalizer', 'compression', 'distortion', 'pitch',
            'formant', 'voice_character', 'chorus', 'flanger', 'reverb'
        ])
        
        if not (0.25 <= speed <= 4.0):
            return jsonify({"error": "Speed must be between 0.25 and 4.0"}), 400
        
        supported_formats = ['mp3', 'wav']
        if response_format not in supported_formats:
            logger.warning(f"Unsupported format '{response_format}', defaulting to mp3")
            response_format = 'mp3'
        
        kokoro_voice_id = get_kokoro_voice(voice_id)
        logger.info(f"Request: voice={kokoro_voice_id}, speed={speed}, format={response_format}, text='{text[:50]}...'")
        
        # Generate base audio
        audio_tensor = enhanced_model_manager.generate_audio(text, kokoro_voice_id, speed, use_gpu_flag)
        audio_numpy = tensor_to_numpy(audio_tensor)
        
        # Apply effects pipeline (only applies effects that are explicitly configured)
        effects_params = data.get('effects', {})
        audio_numpy = apply_effects_pipeline(audio_numpy, effects_params, effect_order)
        
        # Convert to desired format
        audio_bytes = numpy_to_format(audio_numpy, SAMPLE_RATE, response_format)
        mime_type = 'audio/mpeg' if response_format == 'mp3' else f'audio/{response_format}'
        
        return Response(
            audio_bytes,
            mimetype=mime_type,
            headers={
                'Content-Type': mime_type,
                'Content-Length': str(len(audio_bytes)),
                'Access-Control-Allow-Origin': '*',  # For read-aloud compatibility
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            }
        )
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        logger.error(f"RuntimeError: {re}")
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/v1/audio/speech/robust', methods=['POST'])
@check_auth
def create_speech_robust_route():
    """Enhanced speech generation endpoint with robust text processing."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('input', '').strip()
        if not text:
            return jsonify({"error": "Missing or empty 'input' field"}), 400
        
        voice_id = data.get('voice', DEFAULT_VOICE)
        speed = float(data.get('speed', 1.0))
        response_format = data.get('response_format', 'mp3').lower()
        use_gpu_flag = data.get('use_gpu', True)
        max_chunk_length = int(data.get('max_chunk_length', DEFAULT_CHUNK_SIZE))
        silence_between_chunks = float(data.get('silence_between_chunks', 0.2))
        processing_mode = data.get('processing_mode')
        
        # Convert string mode to enum
        if processing_mode:
            try:
                processing_mode = TextProcessingMode(processing_mode.lower())
            except ValueError:
                processing_mode = None
        
        effect_order = data.get('effects', {}).get('order', [
            'volume', 'equalizer', 'compression', 'distortion', 'pitch',
            'formant', 'voice_character', 'chorus', 'flanger', 'reverb'
        ])
        
        if not (0.25 <= speed <= 4.0):
            return jsonify({"error": "Speed must be between 0.25 and 4.0"}), 400
        
        if not (MIN_CHUNK_SIZE <= max_chunk_length <= 1000):
            return jsonify({"error": f"max_chunk_length must be between {MIN_CHUNK_SIZE} and 1000"}), 400
        
        supported_formats = ['mp3', 'wav']
        if response_format not in supported_formats:
            logger.warning(f"Unsupported format '{response_format}', defaulting to mp3")
            response_format = 'mp3'
        
        kokoro_voice_id = get_kokoro_voice(voice_id)
        
        logger.info(f"Robust request: voice={kokoro_voice_id}, speed={speed}, "
                   f"format={response_format}, max_chunk={max_chunk_length}, "
                   f"mode={processing_mode}, text='{text[:50]}...'")
        
        # Generate audio for all chunks using robust processing
        audio_tensors = enhanced_model_manager.generate_audio_robust(
            text, kokoro_voice_id, speed, use_gpu_flag, max_chunk_length, processing_mode
        )
        
        # Concatenate all audio chunks
        combined_audio = enhanced_model_manager.concatenate_audio_tensors(
            audio_tensors, silence_between_chunks
        )
        
        # Convert to numpy and apply effects
        audio_numpy = tensor_to_numpy(combined_audio)
        effects_params = data.get('effects', {})
        audio_numpy = apply_effects_pipeline(audio_numpy, effects_params, effect_order)
        
        # Convert to desired format
        audio_bytes = numpy_to_format(audio_numpy, SAMPLE_RATE, response_format)
        mime_type = 'audio/mpeg' if response_format == 'mp3' else f'audio/{response_format}'
        
        return Response(
            audio_bytes,
            mimetype=mime_type,
            headers={
                'Content-Type': mime_type,
                'Content-Length': str(len(audio_bytes)),
                'X-Chunks-Processed': str(len(audio_tensors)),
                'X-Processing-Mode': processing_mode.value if processing_mode else 'auto',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            }
        )
        
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        logger.error(f"RuntimeError: {re}")
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/v1/audio/speech/stream', methods=['POST'])
@check_auth
def create_speech_stream_route():
    """Streaming speech generation endpoint."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('input', '').strip()
        if not text:
            return jsonify({"error": "Missing or empty 'input' field"}), 400
        
        voice_id = data.get('voice', DEFAULT_VOICE)
        speed = float(data.get('speed', 1.0))
        response_format = data.get('response_format', 'mp3').lower()
        use_gpu_flag = data.get('use_gpu', True)
        max_chunk_length = int(data.get('max_chunk_length', DEFAULT_CHUNK_SIZE))
        processing_mode = data.get('processing_mode')
        
        if processing_mode:
            try:
                processing_mode = TextProcessingMode(processing_mode.lower())
            except ValueError:
                processing_mode = None
        
        if not (0.25 <= speed <= 4.0):
            return jsonify({"error": "Speed must be between 0.25 and 4.0"}), 400
        
        kokoro_voice_id = get_kokoro_voice(voice_id)
        
        def generate_audio_stream():
            try:
                # Process text into chunks
                text_chunks = enhanced_model_manager.text_processor.process_text(
                    text, mode=processing_mode, max_chunk_length=max_chunk_length
                )
                
                effects_params = data.get('effects', {})
                effect_order = effects_params.get('order', [
                    'volume', 'equalizer', 'compression', 'distortion', 'pitch',
                    'formant', 'voice_character', 'chorus', 'flanger', 'reverb'
                ])
                
                for chunk in text_chunks:
                    try:
                        # Generate audio for chunk
                        audio_tensor = enhanced_model_manager.generate_audio_for_chunk(
                            chunk, kokoro_voice_id, speed, use_gpu_flag
                        )
                        
                        # Convert to numpy and apply effects
                        audio_numpy = tensor_to_numpy(audio_tensor)
                        audio_numpy = apply_effects_pipeline(audio_numpy, effects_params, effect_order)
                        
                        # Convert to desired format
                        audio_bytes = numpy_to_format(audio_numpy, SAMPLE_RATE, response_format)
                        
                        yield audio_bytes
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield b''  # Empty response on error
        
        mime_type = 'audio/mpeg' if response_format == 'mp3' else f'audio/{response_format}'
        
        return Response(
            stream_with_context(generate_audio_stream()),
            mimetype=mime_type,
            headers={
                'Content-Type': mime_type,
                'Transfer-Encoding': 'chunked',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming setup error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/v1/audio/speech/play', methods=['POST'])
@check_auth
def play_speech_route():
    """Play speech locally with interrupt capability (Windows only)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('input', '').strip()
        if not text:
            return jsonify({"error": "Missing or empty 'input' field"}), 400
        
        voice_id = data.get('voice', DEFAULT_VOICE)
        speed = float(data.get('speed', 1.0))
        use_gpu_flag = data.get('use_gpu', True)
        use_robust = data.get('use_robust_processing', False)
        
        # Generate unique session ID
        session_id = hashlib.md5(f"{text}_{voice_id}_{speed}_{time.time()}".encode()).hexdigest()[:8]
        
        effect_order = data.get('effects', {}).get('order', [
            'volume', 'equalizer', 'compression', 'distortion', 'pitch',
            'formant', 'voice_character', 'chorus', 'flanger', 'reverb'
        ])
        
        if not (0.25 <= speed <= 4.0):
            return jsonify({"error": "Speed must be between 0.25 and 4.0"}), 400
        
        kokoro_voice_id = get_kokoro_voice(voice_id)
        logger.info(f"Play request: voice={kokoro_voice_id}, speed={speed}, robust={use_robust}, session={session_id}, text='{text[:50]}...'")
        
        if use_robust:
            # Use robust processing
            max_chunk_length = int(data.get('max_chunk_length', DEFAULT_CHUNK_SIZE))
            processing_mode = data.get('processing_mode')
            if processing_mode:
                try:
                    processing_mode = TextProcessingMode(processing_mode.lower())
                except ValueError:
                    processing_mode = None
            
            audio_tensors = enhanced_model_manager.generate_audio_robust(
                text, kokoro_voice_id, speed, use_gpu_flag, max_chunk_length, processing_mode
            )
            combined_audio = enhanced_model_manager.concatenate_audio_tensors(audio_tensors, 0.2)
            audio_numpy = tensor_to_numpy(combined_audio)
        else:
            # Use legacy method
            audio_tensor = enhanced_model_manager.generate_audio(text, kokoro_voice_id, speed, use_gpu_flag)
            audio_numpy = tensor_to_numpy(audio_tensor)
        
        # Apply effects
        effects_params = data.get('effects', {})
        audio_numpy = apply_effects_pipeline(audio_numpy, effects_params, effect_order)
        
        # Start playback in a separate thread to avoid blocking
        def play_in_thread():
            play_audio_windows_with_interrupt(audio_numpy, SAMPLE_RATE, session_id)
        
        playback_thread = threading.Thread(target=play_in_thread, daemon=True)
        playback_thread.start()
        
        return jsonify({
            "status": "success", 
            "message": "Audio playback started",
            "session_id": session_id,
            "robust_processing": use_robust,
            "duration": calculate_audio_duration(audio_numpy, SAMPLE_RATE)
        }), 200
        
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        logger.error(f"RuntimeError: {re}")
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/v1/audio/speech/stop', methods=['POST'])
@check_auth
def stop_speech_route():
    """Stop/interrupt current audio playback."""
    try:
        data = request.get_json()
        session_id = data.get('session_id') if data else None
        
        success = playback_controller.stop_playback(session_id)
        status = playback_controller.get_status()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Audio playback stopped",
                "session_id": status["session_id"],
                "interrupted_at": status["interrupted_at"],
                "total_duration": status["total_duration"]
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "No active playback session or wrong session ID",
                "current_session": status["session_id"]
            }), 400
            
    except Exception as e:
        logger.error(f"Stop playback error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/v1/audio/speech/status', methods=['GET'])
@check_auth
def playback_status_route():
    """Get current playback status."""
    try:
        status = playback_controller.get_status()
        return jsonify({
            "status": "success",
            "playback": status
        }), 200
        
    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/v1/models', methods=['GET'])
@check_auth
def list_models_route():
    return jsonify({
        "object": "list",
        "data": [{
            "id": "kokoro-tts-1",
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "kokoro-project"
        }]
    })

@app.route('/v1/voices', methods=['GET'])
@check_auth
def list_voices_route():
    voices_list = []
    for voice_id, info in KOKORO_VOICES.items():
        voices_list.append({
            "id": voice_id,
            "name": info.get('description', voice_id),
            "gender": info.get('gender', 'unknown'),
            "language_code": info.get('lang', 'unknown'),
            "model_id": "kokoro-tts-1"
        })
    return jsonify({"object": "list", "data": voices_list})

@app.route('/v1/effects', methods=['GET'])
@check_auth
def list_effects_route():
    return jsonify({
        "available_effects": {
            "volume": {
                "gain": {"type": "float", "range": [0.0, 2.0], "default": 1.0, "description": "Linear gain multiplier"},
                "gain_db": {"type": "float", "range": [-60.0, 12.0], "default": None, "description": "Gain in dB, overrides gain"}
            },
            "pitch": {
                "target_note": {
                    "type": "string",
                    "description": "Musical note (e.g., 'C4', 'A#5') or null",
                    "available_notes": available_notes()
                },
                "semitone_shift": {"type": "float", "range": [-12, 12], "default": 0.0, "description": "Pitch shift in semitones"},
                "preserve_formants": {"type": "boolean", "default": False, "description": "Preserve formants during pitch shift"}
            },
            "voice_character": {
                "type": {"type": "string", "options": ["none", "child", "robot", "deep", "whisper", "alien", "monster", "echo"], "default": "none", "description": "Voice character transformation"},
                "params": {
                    "pitch_shift": {"type": "float", "range": [-12, 12], "description": "Custom pitch shift for character"},
                    "formant_shift": {"type": "float", "range": [0.5, 1.5], "description": "Custom formant shift factor"},
                    "speed": {"type": "float", "range": [0.5, 2.0], "description": "Custom time stretch factor"},
                    "carrier_freq": {"type": "float", "range": [50, 200], "description": "Carrier frequency for robot effect"},
                    "distortion_factor": {"type": "float", "range": [0.5, 5.0], "description": "Distortion intensity for robot/monster"},
                    "noise_level": {"type": "float", "range": [0.0, 0.1], "description": "Noise level for whisper"},
                    "signal_level": {"type": "float", "range": [0.0, 1.0], "description": "Signal level for whisper"},
                    "compression_factor": {"type": "float", "range": [0.5, 2.0], "description": "Compression intensity for whisper"},
                    "flanger_delay_ms": {"type": "float", "range": [0.1, 10.0], "description": "Flanger delay for alien"},
                    "distortion_drive": {"type": "float", "range": [0.0, 36.0], "description": "Distortion drive for monster"},
                    "pre_delay_ms": {"type": "float", "range": [0.0, 100.0], "description": "Pre-delay for echo"}
                }
            },
            "equalizer": {
                "bands": {
                    "type": "array",
                    "items": {
                        "frequency_hz": {"type": "float", "range": [20, 20000], "description": "Center frequency in Hz"},
                        "gain_db": {"type": "float", "range": [-24, 24], "description": "Gain in dB"},
                        "q_factor": {"type": "float", "range": [0.1, 10], "default": 1.0, "description": "Bandwidth control"},
                        "type": {"type": "string", "options": ["peak", "low_shelf", "high_shelf"], "description": "Filter type"}
                    }
                }
            },
            "reverb": {
                "room_size_percent": {"type": "float", "range": [0, 100], "default": 0.0, "description": "Room size percentage"},
                "damping_percent": {"type": "float", "range": [0, 100], "default": 50.0, "description": "High frequency damping percentage"},
                "pre_delay_ms": {"type": "float", "range": [0, 100], "default": 0.0, "description": "Pre-delay in milliseconds"},
                "stereo_width": {"type": "float", "range": [0, 1], "default": 0.0, "description": "Stereo width for reverb"}
            },
            "formant": {
                "shift_percent": {"type": "float", "range": [-100, 100], "default": 0.0, "description": "Formant shift percentage"},
                "scale": {"type": "float", "range": [0.5, 2.0], "default": 1.0, "description": "Intensity of formant shift"}
            },
            "distortion": {
                "drive_db": {"type": "float", "range": [0, 36], "default": 0.0, "description": "Drive gain in dB"},
                "type": {"type": "string", "options": ["soft", "hard", "tanh"], "default": "tanh", "description": "Clipping type"},
                "mix": {"type": "float", "range": [0, 1], "default": 0.0, "description": "Dry/wet mix"}
            },
            "chorus": {
                "delay_ms": {"type": "float", "range": [5, 50], "default": 0.0, "description": "Base delay in milliseconds"},
                "depth": {"type": "float", "range": [0, 0.1], "default": 0.0, "description": "Modulation depth"},
                "rate_hz": {"type": "float", "range": [0.1, 5], "default": 0.0, "description": "Modulation rate in Hz"},
                "mix": {"type": "float", "range": [0, 1], "default": 0.0, "description": "Dry/wet mix"}
            },
            "flanger": {
                "delay_ms": {"type": "float", "range": [0.1, 10], "default": 0.0, "description": "Base delay in milliseconds"},
                "depth": {"type": "float", "range": [0, 0.05], "default": 0.0, "description": "Modulation depth"},
                "rate_hz": {"type": "float", "range": [0.1, 10], "default": 0.0, "description": "LFO rate in Hz"},
                "feedback": {"type": "float", "range": [0, 0.9], "default": 0.0, "description": "Feedback amount"},
                "mix": {"type": "float", "range": [0, 1], "default": 0.0, "description": "Dry/wet mix"}
            },
            "compression": {
                "threshold_db": {"type": "float", "range": [-60, 0], "default": 0.0, "description": "Threshold in dB"},
                "ratio": {"type": "float", "range": [1, 20], "default": 1.0, "description": "Compression ratio"},
                "attack_ms": {"type": "float", "range": [0.1, 100], "default": 0.1, "description": "Attack time in milliseconds"},
                "release_ms": {"type": "float", "range": [10, 1000], "default": 10.0, "description": "Release time in milliseconds"}
            },
            "order": {
                "type": "array",
                "items": {"type": "string", "options": ["volume", "equalizer", "compression", "distortion", "pitch", "formant", "voice_character", "chorus", "flanger", "reverb"]},
                "default": ["volume", "equalizer", "compression", "distortion", "pitch", "formant", "voice_character", "chorus", "flanger", "reverb"],
                "description": "Order of effect application"
            }
        }
    })

@app.route('/v1/text/process', methods=['POST'])
@check_auth
def process_text_route():
    """Text processing endpoint for testing and debugging."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('input', '').strip()
        if not text:
            return jsonify({"error": "Missing or empty 'input' field"}), 400
        
        max_chunk_length = int(data.get('max_chunk_length', DEFAULT_CHUNK_SIZE))
        processing_mode = data.get('processing_mode')
        
        if processing_mode:
            try:
                processing_mode = TextProcessingMode(processing_mode.lower())
            except ValueError:
                processing_mode = None
        
        # Process text
        text_chunks = enhanced_model_manager.text_processor.process_text(
            text, mode=processing_mode, max_chunk_length=max_chunk_length
        )
        
        # Format response
        chunks_data = []
        for chunk in text_chunks:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "char_count": chunk.char_count,
                "processing_time": chunk.processing_time
            })
        
        return jsonify({
            "original_text": text,
            "processing_mode": processing_mode.value if processing_mode else "auto",
            "total_chunks": len(text_chunks),
            "chunks": chunks_data
        })
        
    except Exception as e:
        logger.error(f"Text processing error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check_route():
    return jsonify({
        "status": "healthy",
        "cuda_available": CUDA_AVAILABLE,
        "voices_loaded": len(KOKORO_VOICES),
        "default_voice": DEFAULT_VOICE,
        "effects_available": ["volume", "pitch", "voice_character", "equalizer", "reverb", "formant", "distortion", "chorus", "flanger", "compression"],
        "features": {
            "robust_text_processing": True,
            "zero_default_effects": True,
            "playback_control": True,
            "interrupt_capability": True,
            "streaming": True,
            "caching": True,
            "markdown_support": HAS_MARKDOWN,
            "num2words_support": HAS_NUM2WORDS,
            "read_aloud_compatible": True,
            "pitch_shifting_fixed": True
        },
        "cache_stats": {
            "text_cache_size": len(enhanced_model_manager.text_processor._cache),
            "audio_cache_size": len(enhanced_model_manager.audio_cache)
        },
        "playback_status": playback_controller.get_status(),
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    })

@app.route('/', methods=['GET'])
def index_route():
    return jsonify({
        "service": "Complete Kokoro TTS API",
        "version": "3.3.0",
        "description": "Production-grade TTS API with FIXED pitch shifting, read-aloud compatibility, and zero-default effects",
        "status": "running",
        "default_voice": DEFAULT_VOICE,
        "cuda_available": CUDA_AVAILABLE,
        "key_features": {
            "zero_default_effects": "Effects only apply when explicitly configured",
            "robust_text_processing": "Handles markdown, unicode, numbers, abbreviations",
            "playback_control": "Local audio playback with interrupt capability",
            "session_management": "Track and control audio playback sessions",
            "streaming_support": "Real-time audio streaming",
            "divide_by_zero_safe": "All calculations protected against mathematical errors",
            "pitch_shifting_fixed": "Robust pitch shifting with librosa compatibility",
            "read_aloud_compatible": "Full CORS support for browser extensions"
        },
        "endpoints": {
            "speech_generation": "/v1/audio/speech (POST) - Clean zero-default effects",
            "robust_speech_generation": "/v1/audio/speech/robust (POST) - Enhanced with text processing",
            "streaming_speech": "/v1/audio/speech/stream (POST) - Streaming generation",
            "speech_playback": "/v1/audio/speech/play (POST) - Local playback with session control",
            "stop_playback": "/v1/audio/speech/stop (POST) - Stop/interrupt current playback",
            "playback_status": "/v1/audio/speech/status (GET) - Get current playback status",
            "text_processing": "/v1/text/process (POST) - Text processing testing",
            "list_models": "/v1/models (GET)",
            "list_voices": "/v1/voices (GET)",
            "list_effects": "/v1/effects (GET)",
            "health_check": "/health (GET)",
            "ping": "/ping (GET)"
        },
        "example_pitch_requests": {
            "squeaky_voice": {
                "input": "Hello world! This is a test of the squeaky voice effect.",
                "voice": "af_heart",
                "speed": 1.0,
                "effects": {
                    "pitch": {"semitone_shift": 8.0}
                }
            },
            "deep_voice": {
                "input": "Hello world! This is a test of the deep voice effect.",
                "voice": "af_heart", 
                "speed": 1.0,
                "effects": {
                    "pitch": {"semitone_shift": -6.0}
                }
            },
            "child_character": {
                "input": "Hello world! This is a test of the child voice character.",
                "voice": "af_heart",
                "speed": 1.0,
                "effects": {
                    "voice_character": {"type": "child"}
                }
            },
            "monster_character": {
                "input": "Hello world! This is a test of the monster voice character.",
                "voice": "af_heart",
                "speed": 1.0,
                "effects": {
                    "voice_character": {"type": "monster"}
                }
            }
        },
        "read_aloud_compatibility": {
            "cors_enabled": True,
            "supported_formats": ["mp3", "wav"],
            "standard_endpoint": "/v1/audio/speech",
            "example_request": {
                "method": "POST",
                "url": "http://localhost:5000/v1/audio/speech",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "input": "Text to speak",
                    "voice": "af_heart",
                    "response_format": "mp3"
                }
            }
        },
        "playback_features": {
            "interrupt_support": "Stop playback at any time",
            "session_tracking": "Unique session IDs for each playback",
            "timing_info": "Get interrupted time and total duration",
            "status_monitoring": "Real-time playback status"
        }
    })

# Handle OPTIONS requests for CORS preflight
@app.route('/v1/audio/speech', methods=['OPTIONS'])
@app.route('/v1/audio/speech/robust', methods=['OPTIONS'])
@app.route('/v1/audio/speech/stream', methods=['OPTIONS'])
@app.route('/v1/audio/speech/play', methods=['OPTIONS'])
@app.route('/v1/audio/speech/stop', methods=['OPTIONS'])
def handle_options():
    """Handle CORS preflight requests for read-aloud compatibility."""
    return '', 200, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Max-Age': '86400'
    }

# --- Error Handlers ---------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"500 Internal Server Error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

# --- Main -------------------------------------------------------------------

def main():
    logger.info("=" * 80)
    logger.info("Complete Kokoro TTS API Server v3.3.0 - FIXED EDITION")
    logger.info("=" * 80)
    logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
    logger.info(f"Total Voices: {len(KOKORO_VOICES)}")
    logger.info(f"Default Voice: {DEFAULT_VOICE}")
    logger.info(f"CORS Enabled: {ENABLE_CORS}")
    logger.info(f"Server Address: http://{API_HOST}:{API_PORT}")
    logger.info(f"Markdown Support: {HAS_MARKDOWN}")
    logger.info(f"Num2Words Support: {HAS_NUM2WORDS}")
    logger.info("Key Features:")
    logger.info("  ✓ Zero-default effects (clean slate approach)")
    logger.info("  ✓ Playback control with interrupt capability")
    logger.info("  ✓ Session management and status tracking")
    logger.info("  ✓ Robust text processing and streaming")
    logger.info("  ✓ Divide-by-zero safe calculations")
    logger.info("  ✓ FIXED pitch shifting with librosa compatibility")
    logger.info("  ✓ Full read-aloud extension compatibility")
    logger.info("=" * 80)
    logger.info("PITCH SHIFTING EXAMPLES:")
    logger.info("  Squeaky: {'effects': {'pitch': {'semitone_shift': 8.0}}}")
    logger.info("  Deep: {'effects': {'pitch': {'semitone_shift': -6.0}}}")
    logger.info("  Child: {'effects': {'voice_character': {'type': 'child'}}}")
    logger.info("  Monster: {'effects': {'voice_character': {'type': 'monster'}}}")
    logger.info("=" * 80)
    app.run(host=API_HOST, port=API_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
