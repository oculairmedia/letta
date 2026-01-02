import ipaddress
import socket
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from letta.log import get_logger

logger = get_logger(__name__)


PRIVATE_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

DEFAULT_BLOCKED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "169.254.169.254",
]


def is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in network for network in PRIVATE_IP_RANGES)
    except ValueError:
        return False


def resolve_hostname(hostname: str) -> List[str]:
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return list({str(result[4][0]) for result in results})
    except socket.gaierror:
        return []


def validate_webhook_url(
    url: str,
    blocked_hosts: Optional[List[str]] = None,
    allowed_hosts: Optional[List[str]] = None,
    require_https: bool = False,
) -> Tuple[bool, Optional[str]]:
    if blocked_hosts is None:
        blocked_hosts = DEFAULT_BLOCKED_HOSTS

    if not url:
        return False, "URL cannot be empty"

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"

    if not parsed.scheme:
        return False, "URL must include a scheme (http:// or https://)"

    if parsed.scheme not in ("http", "https"):
        return False, f"Invalid URL scheme: {parsed.scheme}. Only http and https are allowed"

    if require_https and parsed.scheme != "https":
        return False, "URL must use HTTPS"

    if not parsed.netloc:
        return False, "URL must include a host"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL must include a valid hostname"

    if allowed_hosts:
        if hostname not in allowed_hosts:
            resolved_ips = resolve_hostname(hostname)
            if not any(ip in allowed_hosts for ip in resolved_ips):
                return False, f"Host '{hostname}' is not in the allowed hosts list"
        return True, None

    hostname_lower = hostname.lower()
    for blocked in blocked_hosts:
        if hostname_lower == blocked.lower():
            return False, f"Host '{hostname}' is blocked"

    if is_private_ip(hostname):
        return False, f"IP address '{hostname}' is in a private range"

    resolved_ips = resolve_hostname(hostname)
    if not resolved_ips:
        logger.warning(f"Could not resolve hostname '{hostname}' for SSRF validation")
        return True, None

    for ip in resolved_ips:
        if is_private_ip(ip):
            return False, f"Hostname '{hostname}' resolves to private IP '{ip}'"

    return True, None


def validate_webhook_url_strict(url: str) -> Tuple[bool, Optional[str]]:
    return validate_webhook_url(url, require_https=True)
