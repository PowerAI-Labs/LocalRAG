# api/rate_limiter.py

from fastapi import HTTPException, Request
from collections import defaultdict
import time
from ..core import settings

# Store for rate limiting data
RATE_LIMIT_STORE = defaultdict(list)  # {ip_address: [timestamp1, timestamp2, ...]}

async def check_rate_limit(request: Request):
    """
    Rate limiting middleware.
    
    Limits requests based on client IP address. Default limits are:
    - 100 requests per minute per IP
    
    Raises:
        HTTPException: If rate limit is exceeded
    """
    client_ip = request.client.host
    now = time.time()
    
    # Clean old requests
    RATE_LIMIT_STORE[client_ip] = [
        timestamp for timestamp in RATE_LIMIT_STORE[client_ip]
        if timestamp > now - settings.RATE_LIMIT_DURATION
    ]
    
    # Check if rate limit is exceeded
    if len(RATE_LIMIT_STORE[client_ip]) >= settings.RATE_LIMIT_MAX_REQUESTS:
        # Calculate time until next available request
        oldest_timestamp = min(RATE_LIMIT_STORE[client_ip])
        reset_time = oldest_timestamp + settings.RATE_LIMIT_DURATION
        wait_time = int(reset_time - now)
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Too many requests",
                "wait_seconds": wait_time,
                "reset_time": reset_time
            }
        )
    
    # Add current request timestamp
    RATE_LIMIT_STORE[client_ip].append(now)

class RateLimiter:
    """Class-based rate limiter for more complex rate limiting scenarios."""
    
    def __init__(self):
        self.store = defaultdict(list)
        self.duration = settings.RATE_LIMIT_DURATION
        self.max_requests = settings.RATE_LIMIT_MAX_REQUESTS
    
    def is_rate_limited(self, key: str) -> bool:
        """Check if a key is rate limited."""
        now = time.time()
        
        # Clean old requests
        self.store[key] = [
            ts for ts in self.store[key]
            if ts > now - self.duration
        ]
        
        return len(self.store[key]) >= self.max_requests
    
    def add_request(self, key: str):
        """Add a request for a key."""
        self.store[key].append(time.time())
    
    def get_reset_time(self, key: str) -> float:
        """Get reset time for a rate-limited key."""
        if not self.store[key]:
            return 0
            
        oldest_timestamp = min(self.store[key])
        return oldest_timestamp + self.duration
    
    def clear(self, key: str):
        """Clear rate limiting data for a key."""
        if key in self.store:
            del self.store[key]