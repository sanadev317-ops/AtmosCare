# ============================================================================
# ERROR HANDLING & RESILIENCE LAYER
# ============================================================================
"""
Comprehensive error handling and fallback strategies for Kivy↔Backend integration.
Ensures system remains stable even when API is unavailable.
"""

import logging
from typing import Optional, Callable, Any, Dict
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """System connection states."""
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class ErrorRecoveryStrategy:
    """
    Handles errors with exponential backoff and failover strategies.
    """
    
    def __init__(self, 
                 max_retries: int = 5,
                 initial_backoff: float = 1.0,
                 max_backoff: float = 60.0):
        """
        Initialize recovery strategy.
        
        Args:
            max_retries: Maximum retry attempts
            initial_backoff: Initial backoff in seconds
            max_backoff: Maximum backoff in seconds
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        
        # State tracking
        self.retry_count = 0
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.state = ConnectionState.CONNECTED
        
        # Callbacks
        self.on_state_change: Optional[Callable[[ConnectionState], None]] = None
        self.on_recovery: Optional[Callable[[], None]] = None
    
    def calculate_backoff(self) -> float:
        """Calculate exponential backoff time."""
        backoff = self.initial_backoff * (2 ** self.retry_count)
        return min(backoff, self.max_backoff)
    
    def record_error(self, error: str, state: ConnectionState = ConnectionState.RECONNECTING):
        """
        Record an error and update state.
        
        Args:
            error: Error message
            state: New connection state
        """
        self.retry_count += 1
        self.last_error = error
        self.last_error_time = datetime.now()
        
        # Update state if different
        if state != self.state:
            old_state = self.state
            self.state = state
            
            if self.on_state_change:
                self.on_state_change(state)
            
            logger.warning(
                f"State transition: {old_state.value} → {state.value} | "
                f"Error: {error} | Retry: {self.retry_count}/{self.max_retries}"
            )
    
    def record_success(self):
        """Record successful operation and reset retry counter."""
        if self.retry_count > 0:
            logger.info(f"Recovery successful after {self.retry_count} retries")
            if self.on_recovery:
                self.on_recovery()
        
        self.retry_count = 0
        self.last_error = None
        
        # Restore connected state if degraded
        if self.state != ConnectionState.CONNECTED:
            self.state = ConnectionState.CONNECTED
            if self.on_state_change:
                self.on_state_change(self.state)
    
    def should_retry(self) -> bool:
        """Check if we should retry."""
        return self.retry_count <= self.max_retries
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.state == ConnectionState.CONNECTED and self.retry_count == 0


class FallbackManager:
    """
    Manages fallback data sources when primary backend is unavailable.
    """
    
    def __init__(self):
        """Initialize fallback manager."""
        self.cached_data: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(minutes=30)  # Cache valid for 30 min
        
        # Fallback data providers
        self.fallback_providers: Dict[str, Callable] = {}
    
    def cache_data(self, key: str, data: Any):
        """Cache data for fallback use."""
        self.cached_data[key] = data
        self.cache_timestamps[key] = datetime.now()
        logger.debug(f"Cached data: {key}")
    
    def get_cached_data(self, key: str, allow_stale: bool = False) -> Optional[Any]:
        """
        Retrieve cached data.
        
        Args:
            key: Cache key
            allow_stale: Allow stale data (older than TTL)
        
        Returns:
            Cached data or None if not found/expired
        """
        if key not in self.cached_data:
            return None
        
        # Check TTL
        age = datetime.now() - self.cache_timestamps[key]
        if age > self.cache_ttl and not allow_stale:
            logger.warning(f"Cache expired for: {key} (age: {age})")
            return None
        
        logger.debug(f"Retrieved cached data: {key} (age: {age})")
        return self.cached_data[key]
    
    def register_fallback_provider(self, key: str, provider: Callable):
        """
        Register a fallback data provider function.
        
        Args:
            key: Data key
            provider: Callable that returns fallback data
        """
        self.fallback_providers[key] = provider
        logger.info(f"Registered fallback provider: {key}")
    
    def get_fallback_data(self, key: str) -> Optional[Any]:
        """
        Get fallback data from cache or provider.
        
        Args:
            key: Data key
        
        Returns:
            Fallback data or None
        """
        # Try cache first
        cached = self.get_cached_data(key, allow_stale=True)
        if cached is not None:
            logger.info(f"Using cached fallback: {key}")
            return cached
        
        # Try provider function
        if key in self.fallback_providers:
            try:
                provider_data = self.fallback_providers[key]()
                logger.info(f"Using provider fallback: {key}")
                return provider_data
            except Exception as e:
                logger.warning(f"Fallback provider failed ({key}): {e}")
        
        return None
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cached_data.clear()
        self.cache_timestamps.clear()
        logger.info("Cache cleared")


class ErrorHandler:
    """
    Central error handling for the integration layer.
    """
    
    def __init__(self):
        """Initialize error handler."""
        self.recovery = ErrorRecoveryStrategy()
        self.fallback = FallbackManager()
        
        # Error callbacks
        self.on_critical_error: Optional[Callable[[str], None]] = None
        self.on_warning: Optional[Callable[[str], None]] = None
    
    def handle_connection_error(self, error: str) -> bool:
        """
        Handle connection errors with recovery strategy.
        
        Args:
            error: Error message
        
        Returns:
            True if should retry, False if max retries exceeded
        """
        self.recovery.record_error(error, ConnectionState.RECONNECTING)
        
        if not self.recovery.should_retry():
            self.recovery.record_error(error, ConnectionState.OFFLINE)
            if self.on_critical_error:
                self.on_critical_error(
                    f"Backend unreachable after {self.recovery.retry_count} attempts. "
                    f"Using cached/fallback data."
                )
            return False
        
        if self.on_warning:
            self.on_warning(
                f"Connection attempt {self.recovery.retry_count}/{self.recovery.max_retries}: {error}"
            )
        
        return True
    
    def handle_validation_error(self, error: str):
        """Handle data validation errors."""
        logger.warning(f"Validation error: {error}")
        if self.on_warning:
            self.on_warning(f"Invalid data received: {error}")
    
    def handle_parsing_error(self, error: str, data: Any):
        """Handle JSON/data parsing errors."""
        logger.error(f"Parsing error: {error} | Data: {data}")
        if self.on_warning:
            self.on_warning(f"Failed to parse server response: {error}")
    
    def record_success(self):
        """Record successful operation."""
        self.recovery.record_success()


# ─────────────────────────────────────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────────────────────────────────

_error_handler: Optional[ErrorHandler] = None


def init_error_handler() -> ErrorHandler:
    """Initialize global error handler."""
    global _error_handler
    _error_handler = ErrorHandler()
    return _error_handler


def get_error_handler() -> ErrorHandler:
    """Get global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler
