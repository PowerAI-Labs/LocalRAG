import logging
import sys
import traceback
from datetime import datetime

class ConsoleLogger:
    """Direct console logger with colored output and exception support"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    def __init__(self, name='rag_app'):
        self.name = name

    def _log(self, level: str, color: str, message: str, exc_info=None):
        """Enhanced log method with exception support."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{color}[{timestamp}] {level}: {message}{self.RESET}", flush=True)
        
        if exc_info:
            if isinstance(exc_info, bool) and exc_info:
                # Get current exception info
                exc_type, exc_value, exc_traceback = sys.exc_info()
            elif isinstance(exc_info, tuple):
                exc_type, exc_value, exc_traceback = exc_info
            else:
                return
                
            if exc_traceback:
                # Format exception traceback
                tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                print(f"{color}{''.join(tb_lines)}{self.RESET}", flush=True)

    def info(self, message: str, exc_info=None):
        """Log info message."""
        self._log('INFO', self.BLUE, message, exc_info)

    def warning(self, message: str, exc_info=None):
        """Log warning message."""
        self._log('WARNING', self.YELLOW, message, exc_info)

    def error(self, message: str, exc_info=None):
        """Log error message with optional exception info."""
        self._log('ERROR', self.RED, message, exc_info)

    def debug(self, message: str, exc_info=None):
        """Log debug message."""
        self._log('DEBUG', self.GREEN, message, exc_info)

def setup_logging():
    """Set up logging configuration"""
    console_logger = ConsoleLogger()
    console_logger.info("Logging system initialized")
    return console_logger

# Create logger instance
logger = ConsoleLogger()