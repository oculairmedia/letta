# Issue: Add webhook system settings and configuration

**Type:** Enhancement  
**Priority:** Medium  
**Labels:** `enhancement`, `webhooks`, `configuration`

## Overview
Add global webhook system settings to control timeout, retries, security, and delivery behavior. This provides system-wide defaults that individual agent configurations can override.

## Scope
Extend `letta/settings.py` with webhook-specific configuration options and create validation rules for webhook URLs.

## Settings to Add

### Global Webhook Settings (`letta/settings.py`)
```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Webhook System Settings
    webhook_enabled: bool = Field(
        default=True,
        description="Master switch to enable/disable webhook delivery system-wide"
    )
    
    webhook_timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=30,
        description="HTTP timeout for webhook delivery requests"
    )
    
    webhook_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts for failed webhooks"
    )
    
    webhook_retry_backoff_base: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Base multiplier for exponential backoff (retry_delay = base^attempt)"
    )
    
    webhook_max_delivery_age_hours: int = Field(
        default=24,
        ge=1,
        description="Maximum age of webhook delivery attempts before cleanup"
    )
    
    # Security Settings
    webhook_allow_private_ips: bool = Field(
        default=True,
        description="Allow webhooks to private IP ranges (disable in production)"
    )
    
    webhook_allowed_hosts: Optional[List[str]] = Field(
        default=None,
        description="Allowlist of host patterns for webhook URLs (e.g., ['*.example.com'])"
    )
    
    webhook_blocked_hosts: Optional[List[str]] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "0.0.0.0"],
        description="Blocklist of host patterns to reject"
    )
    
    # Rate Limiting
    webhook_rate_limit_per_agent: int = Field(
        default=100,
        ge=1,
        description="Max webhooks per agent per minute"
    )
    
    # Async Delivery (Redis)
    webhook_use_redis_queue: bool = Field(
        default=True,
        description="Use Redis queue for async webhook delivery (recommended)"
    )
    
    webhook_redis_stream_name: str = Field(
        default="letta:webhooks:events",
        description="Redis stream name for webhook events"
    )
    
    webhook_worker_batch_size: int = Field(
        default=10,
        description="Number of webhooks to process per batch in background worker"
    )
```

## URL Validation Utility

### Create `letta/utils/webhook_validation.py`
```python
from urllib.parse import urlparse
from typing import List, Optional
import ipaddress
import fnmatch

class WebhookURLValidator:
    """Validates webhook URLs against security policies"""
    
    def __init__(
        self,
        allow_private_ips: bool = True,
        allowed_hosts: Optional[List[str]] = None,
        blocked_hosts: Optional[List[str]] = None
    ):
        self.allow_private_ips = allow_private_ips
        self.allowed_hosts = allowed_hosts or []
        self.blocked_hosts = blocked_hosts or []
    
    def validate(self, url: str) -> tuple[bool, Optional[str]]:
        """
        Validate webhook URL against security policies.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)
            
            # Must be HTTP/HTTPS
            if parsed.scheme not in ("http", "https"):
                return False, "Webhook URL must use HTTP or HTTPS"
            
            # Check blocklist
            if self._is_blocked(parsed.hostname):
                return False, f"Host {parsed.hostname} is blocked"
            
            # Check allowlist (if configured)
            if self.allowed_hosts and not self._is_allowed(parsed.hostname):
                return False, f"Host {parsed.hostname} not in allowlist"
            
            # Check private IP ranges
            if not self.allow_private_ips and self._is_private_ip(parsed.hostname):
                return False, "Private IP addresses not allowed"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid URL: {str(e)}"
    
    def _is_blocked(self, hostname: str) -> bool:
        """Check if hostname matches blocklist patterns"""
        return any(fnmatch.fnmatch(hostname, pattern) for pattern in self.blocked_hosts)
    
    def _is_allowed(self, hostname: str) -> bool:
        """Check if hostname matches allowlist patterns"""
        return any(fnmatch.fnmatch(hostname, pattern) for pattern in self.allowed_hosts)
    
    def _is_private_ip(self, hostname: str) -> bool:
        """Check if hostname resolves to private IP range"""
        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            # Not an IP address, assume public
            return False
```

## Environment Variable Support

### Example `.env` Configuration
```bash
# Webhook System
LETTA_WEBHOOK_ENABLED=true
LETTA_WEBHOOK_TIMEOUT_SECONDS=5
LETTA_WEBHOOK_MAX_RETRIES=3
LETTA_WEBHOOK_RETRY_BACKOFF_BASE=2

# Security
LETTA_WEBHOOK_ALLOW_PRIVATE_IPS=false
LETTA_WEBHOOK_ALLOWED_HOSTS='["*.example.com","*.mycompany.com"]'
LETTA_WEBHOOK_BLOCKED_HOSTS='["localhost","127.0.0.1"]'

# Redis Queue
LETTA_WEBHOOK_USE_REDIS_QUEUE=true
LETTA_WEBHOOK_REDIS_STREAM_NAME=letta:webhooks:events
```

## Acceptance Criteria
- [ ] Webhook settings added to `Settings` class with validation
- [ ] Environment variable support for all webhook settings
- [ ] `WebhookURLValidator` utility created with SSRF protections
- [ ] Unit tests for URL validation (private IPs, blocklist, allowlist)
- [ ] Documentation for webhook configuration options
- [ ] Default values are safe for production use

## Security Considerations

### SSRF Protection
- Block localhost/loopback by default
- Optional allowlist for production environments
- DNS rebinding protection (resolve at dispatch time)
- Timeout enforcement to prevent slow loris attacks

### Secrets Management
- Webhook secrets should use secure random generation
- Consider integration with secrets management (Vault, AWS Secrets Manager)
- Secrets should never appear in logs

### Rate Limiting
- Per-agent rate limits prevent abuse
- Consider global rate limit for entire system
- Failed webhooks count toward rate limit

## Files to Create/Modify
- `letta/settings.py` (modify)
- `letta/utils/webhook_validation.py` (create)
- `tests/test_webhook_validation.py` (create)
- `docs/webhooks/configuration.md` (create - user documentation)

## Related Issues
- `01-webhook-module-foundation.md` - WebhookManager uses these settings
- `02-orm-model-webhook-fields.md` - Agent-level overrides for settings
- `06-redis-async-delivery.md` - Redis queue settings used here

## References
- Existing settings: `letta/settings.py`
- Redis settings: `letta/settings.py:258-259`
- Similar URL validation patterns in other projects (Django, Rails)
