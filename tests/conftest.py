"""
Shared test fixtures and configuration for the Spotify LLM Controller tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import json


@pytest.fixture
def mock_mcp_session():
    """Create a mock MCP session for testing."""
    session = Mock()
    session.call_tool = AsyncMock()
    session.initialize = AsyncMock()
    return session


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    client = Mock()
    client.create_completion = Mock()
    client.api_key = "test-api-key"
    client.openai_version = "1.5.0"
    return client


@pytest.fixture
def sample_search_result():
    """Sample search result for testing."""
    return {
        "uri": "spotify:track:track123",
        "id": "track123",
        "name": "Test Song",
        "artist": "Test Artist",
        "type": "track"
    }


@pytest.fixture
def sample_album_search_result():
    """Sample album search result for testing."""
    return {
        "uri": "spotify:album:album123",
        "id": "album123",
        "name": "Test Album",
        "artist": "Test Artist",
        "type": "album"
    }


@pytest.fixture
def mock_mcp_result_success():
    """Create a mock successful MCP result."""
    result = Mock()
    result.isError = False
    result.content = [Mock(text='{"success": true}')]
    return result


@pytest.fixture
def mock_mcp_result_error():
    """Create a mock error MCP result."""
    result = Mock()
    result.isError = True
    result.content = [Mock(text='{"error": "Something went wrong"}')]
    return result


@pytest.fixture
def mock_track_search_response():
    """Mock response for track search."""
    return json.dumps({
        "tracks": [{
            "id": "track123",
            "name": "Test Song",
            "artists": ["Test Artist"]
        }]
    })


@pytest.fixture
def mock_album_search_response():
    """Mock response for album search."""
    return json.dumps({
        "albums": [{
            "id": "album123",
            "name": "Test Album",
            "artists": ["Test Artist"]
        }]
    })


@pytest.fixture
def mock_playlist_response():
    """Mock response for playlist operations."""
    return json.dumps([
        {"id": "playlist1", "name": "My Playlist 1"},
        {"id": "playlist2", "name": "My Playlist 2"}
    ])


@pytest.fixture
def sample_actions_sequence():
    """Sample sequence of actions for testing."""
    return [
        {
            "tool_name": "SpotifySearch",
            "params": {
                "query": "test song",
                "qtype": "track",
                "limit": 1
            }
        },
        {
            "tool_name": "SpotifyPlayback",
            "params": {
                "action": "start"
            }
        }
    ]


@pytest.fixture
def sample_queue_actions_sequence():
    """Sample sequence of actions for queueing."""
    return [
        {
            "tool_name": "SpotifySearch",
            "params": {
                "query": "test song",
                "qtype": "track",
                "limit": 1
            }
        },
        {
            "tool_name": "SpotifyQueue",
            "params": {
                "action": "add"
            }
        }
    ]


@pytest.fixture
def mock_context():
    """Mock execution context for testing."""
    return {
        "search_result": None,
        "album_tracks": None
    }


@pytest.fixture
def mock_context_with_search():
    """Mock execution context with search result."""
    return {
        "search_result": {
            "uri": "spotify:track:track123",
            "id": "track123",
            "name": "Test Song",
            "artist": "Test Artist",
            "type": "track"
        },
        "album_tracks": None
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "MCP_SERVER_URL": "http://test-server:8080",
        "MCP_CLIENT_PORT": "9090",
        "OPENAI_API_KEY": "test-api-key",
        "OPENAI_MODEL": "gpt-4",
        "OPENAI_MAX_TOKENS": "200"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


class MockStreamableClient:
    """Mock streamable HTTP client for testing."""
    
    def __init__(self, url):
        self.url = url
    
    async def __aenter__(self):
        return (Mock(), Mock(), Mock())  # read, write, close
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockClientSession:
    """Mock MCP client session for testing."""
    
    def __init__(self, read, write):
        self.read = read
        self.write = write
        self.call_tool = AsyncMock()
        self.initialize = AsyncMock()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_streamable_client():
    """Mock streamable client fixture."""
    return MockStreamableClient


@pytest.fixture
def mock_client_session():
    """Mock client session fixture."""
    return MockClientSession