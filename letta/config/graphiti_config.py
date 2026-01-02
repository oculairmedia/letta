# graphiti_config.py
"""
Configuration for Graphiti integration with Letta.
Controls whether to use Graphiti for archival memory and related settings.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti integration"""
    
    # Enable/disable Graphiti integration
    enabled: bool = Field(
        default=False,
        description="Enable Graphiti for archival memory"
    )
    
    # Graphiti API endpoint
    url: str = Field(
        default="http://192.168.50.90:8001",
        description="Graphiti API URL"
    )
    
    # Timeout for Graphiti requests
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    
    # Entity extraction settings
    extract_entities: bool = Field(
        default=True,
        description="Automatically extract entities from memories"
    )
    
    # Relationship extraction
    extract_relationships: bool = Field(
        default=True,
        description="Automatically extract relationships between entities"
    )
    
    # Batch settings
    batch_size: int = Field(
        default=10,
        description="Batch size for bulk operations"
    )
    
    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable local caching of search results"
    )
    
    cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds"
    )
    
    @classmethod
    def from_env(cls) -> "GraphitiConfig":
        """Load configuration from environment variables"""
        return cls(
            enabled=os.getenv("LETTA_GRAPHITI_ENABLED", "false").lower() == "true",
            url=os.getenv("LETTA_GRAPHITI_URL", "http://192.168.50.90:8001"),
            timeout=int(os.getenv("LETTA_GRAPHITI_TIMEOUT", "30")),
            extract_entities=os.getenv("LETTA_GRAPHITI_EXTRACT_ENTITIES", "true").lower() == "true",
            extract_relationships=os.getenv("LETTA_GRAPHITI_EXTRACT_RELATIONSHIPS", "true").lower() == "true",
            batch_size=int(os.getenv("LETTA_GRAPHITI_BATCH_SIZE", "10")),
            cache_enabled=os.getenv("LETTA_GRAPHITI_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("LETTA_GRAPHITI_CACHE_TTL", "300"))
        )


# Global instance
graphiti_config = GraphitiConfig.from_env()


def is_graphiti_enabled() -> bool:
    """Check if Graphiti integration is enabled"""
    return graphiti_config.enabled


def get_graphiti_url() -> str:
    """Get configured Graphiti URL"""
    return graphiti_config.url


def enable_graphiti(url: Optional[str] = None):
    """Enable Graphiti integration programmatically"""
    global graphiti_config
    graphiti_config.enabled = True
    if url:
        graphiti_config.url = url
    
    # Also set environment variable for consistency
    os.environ["LETTA_GRAPHITI_ENABLED"] = "true"
    if url:
        os.environ["LETTA_GRAPHITI_URL"] = url


def disable_graphiti():
    """Disable Graphiti integration"""
    global graphiti_config
    graphiti_config.enabled = False
    os.environ["LETTA_GRAPHITI_ENABLED"] = "false"