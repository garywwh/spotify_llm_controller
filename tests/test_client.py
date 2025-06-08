"""
Unit tests for the client module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from src.spotify_llm_controller.client import app, parse_command_with_llm, CommandRequest


class TestParseCommandWithLLM:
    """Test the LLM command parsing functionality."""
    
    def test_successful_parsing(self):
        """Test successful command parsing."""
        mock_openai_client = Mock()
        mock_openai_client.create_completion.return_value = '[{"tool_name": "SpotifySearch", "params": {"query": "test"}}]'
        
        with patch('src.spotify_llm_controller.client.OpenAIClient', return_value=mock_openai_client), \
             patch('src.spotify_llm_controller.client.parse_llm_response', return_value=[{"tool_name": "SpotifySearch", "params": {"query": "test"}}]):
            result = parse_command_with_llm("play test song")
        
        assert isinstance(result, list)
        assert result[0]["tool_name"] == "SpotifySearch"
        assert result[0]["params"]["query"] == "test"
    
    def test_openai_client_failure(self):
        """Test handling of OpenAI client failure."""
        with patch('src.spotify_llm_controller.client.OpenAIClient', side_effect=Exception("API Error")):
            result = parse_command_with_llm("play test song")
        
        assert "error" in result
        assert "Failed to parse command" in result["error"]
    
    def test_llm_response_parsing_failure(self):
        """Test handling of LLM response parsing failure."""
        mock_openai_client = Mock()
        mock_openai_client.create_completion.return_value = "invalid response"
        
        with patch('src.spotify_llm_controller.client.OpenAIClient', return_value=mock_openai_client), \
             patch('src.spotify_llm_controller.client.parse_llm_response', side_effect=ValueError("Parse error")):
            result = parse_command_with_llm("play test song")
        
        assert "error" in result
        assert "Failed to parse command" in result["error"]
    
    def test_prompt_formatting(self):
        """Test that the prompt is properly formatted with the command."""
        mock_openai_client = Mock()
        mock_openai_client.create_completion.return_value = '[]'
        
        with patch('src.spotify_llm_controller.client.OpenAIClient', return_value=mock_openai_client), \
             patch('src.spotify_llm_controller.client.parse_llm_response', return_value=[]), \
             patch('src.spotify_llm_controller.config.SPOTIFY_COMMAND_PROMPT', 'Test prompt: {command}'):
            parse_command_with_llm("test command")
        
        # Verify the prompt was formatted with the command
        call_args = mock_openai_client.create_completion.call_args[0][0]
        assert "Test prompt: test command" == call_args


class TestFastAPIEndpoints:
    """Test the FastAPI endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Spotify MCP Client"
        assert "endpoints" in data
        assert "/command" in data["endpoints"]
        assert "/health" in data["endpoints"]
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "spotify-mcp-client"
        assert data["version"] == "1.0.0"
    
    @patch('src.spotify_llm_controller.client.parse_command_with_llm')
    @patch('src.spotify_llm_controller.client.execute_spotify_actions')
    @patch('src.spotify_llm_controller.client.streamablehttp_client')
    @patch('src.spotify_llm_controller.client.ClientSession')
    def test_command_endpoint_success(self, mock_client_session, mock_streamable_client, 
                                     mock_execute_actions, mock_parse_command):
        """Test successful command execution."""
        # Setup mocks
        mock_parse_command.return_value = [{"tool_name": "SpotifySearch", "params": {"query": "test"}}]
        mock_execute_actions.return_value = {"message": "Success"}
        
        # Mock the async context managers
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_streamable_client.return_value.__aenter__ = AsyncMock(return_value=(None, None, None))
        mock_streamable_client.return_value.__aexit__ = AsyncMock(return_value=None)
        
        response = self.client.post("/command", json={"command": "play test song"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Success"
    
    @patch('src.spotify_llm_controller.client.parse_command_with_llm')
    def test_command_endpoint_parse_error(self, mock_parse_command):
        """Test command endpoint with parsing error."""
        mock_parse_command.return_value = {"error": "Parse failed"}
        
        response = self.client.post("/command", json={"command": "invalid command"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "Parse failed"
    
    @patch('src.spotify_llm_controller.client.parse_command_with_llm')
    @patch('src.spotify_llm_controller.client.streamablehttp_client')
    def test_command_endpoint_mcp_connection_error(self, mock_streamable_client, mock_parse_command):
        """Test command endpoint with MCP connection error."""
        mock_parse_command.return_value = [{"tool_name": "SpotifySearch", "params": {"query": "test"}}]
        mock_streamable_client.side_effect = Exception("Connection failed")
        
        response = self.client.post("/command", json={"command": "play test song"})
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Connection failed" in data["error"]
    
    def test_command_endpoint_invalid_request(self):
        """Test command endpoint with invalid request format."""
        response = self.client.post("/command", json={"invalid": "field"})
        
        assert response.status_code == 422  # Validation error
    
    def test_command_endpoint_missing_command(self):
        """Test command endpoint with missing command field."""
        response = self.client.post("/command", json={})
        
        assert response.status_code == 422  # Validation error


class TestCommandRequest:
    """Test the CommandRequest model."""
    
    def test_valid_command_request(self):
        """Test creating a valid CommandRequest."""
        request = CommandRequest(command="play test song")
        assert request.command == "play test song"
    
    def test_command_request_validation(self):
        """Test CommandRequest validation."""
        # Test that command is required
        with pytest.raises(ValueError):
            CommandRequest()


class TestIntegration:
    """Integration tests for the client module."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('src.spotify_llm_controller.client.OpenAIClient')
    @patch('src.spotify_llm_controller.client.execute_spotify_actions')
    @patch('src.spotify_llm_controller.client.streamablehttp_client')
    @patch('src.spotify_llm_controller.client.ClientSession')
    def test_full_command_flow(self, mock_client_session, mock_streamable_client, 
                              mock_execute_actions, mock_openai_client):
        """Test the full command processing flow."""
        # Setup OpenAI mock
        mock_openai_instance = Mock()
        mock_openai_instance.create_completion.return_value = '''[
            {"tool_name": "SpotifySearch", "params": {"query": "test song", "qtype": "track", "limit": 1}},
            {"tool_name": "SpotifyPlayback", "params": {"action": "start"}}
        ]'''
        mock_openai_client.return_value = mock_openai_instance
        
        # Setup MCP session mock
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_streamable_client.return_value.__aenter__ = AsyncMock(return_value=(None, None, None))
        mock_streamable_client.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Setup action execution mock
        mock_execute_actions.return_value = {
            "message": "Playing track test song by test artist",
            "details": {"action": "playback", "type": "start"}
        }
        
        response = self.client.post("/command", json={"command": "play test song"})
        
        assert response.status_code == 200
        data = response.json()
        assert "Playing track test song" in data["message"]
        
        # Verify the flow
        mock_openai_instance.create_completion.assert_called_once()
        mock_execute_actions.assert_called_once()
        
        # Verify the parsed actions were passed correctly
        call_args = mock_execute_actions.call_args[0][1]  # Second argument is actions
        assert len(call_args) == 2
        assert call_args[0]["tool_name"] == "SpotifySearch"
        assert call_args[1]["tool_name"] == "SpotifyPlayback"
    
    @patch('src.spotify_llm_controller.client.OpenAIClient')
    def test_openai_initialization_error(self, mock_openai_client):
        """Test handling of OpenAI initialization error."""
        mock_openai_client.side_effect = ValueError("OpenAI API key is required")
        
        response = self.client.post("/command", json={"command": "play test song"})
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Failed to parse command" in data["error"]
    
    def test_command_logging(self, caplog):
        """Test that commands are properly logged."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch('src.spotify_llm_controller.client.parse_command_with_llm', return_value={"error": "test"}):
            self.client.post("/command", json={"command": "test command"})
        
        assert "Received command: test command" in caplog.text


class TestConfigurationIntegration:
    """Test integration with configuration."""
    
    @patch('src.spotify_llm_controller.client.MCP_SERVER_URL', 'http://test-server:8080')
    @patch('src.spotify_llm_controller.client.parse_command_with_llm')
    @patch('src.spotify_llm_controller.client.streamablehttp_client')
    def test_mcp_server_url_usage(self, mock_streamable_client, mock_parse_command):
        """Test that the correct MCP server URL is used."""
        mock_parse_command.return_value = [{"tool_name": "SpotifySearch", "params": {"query": "test"}}]
        mock_streamable_client.side_effect = Exception("Connection test")
        
        client = TestClient(app)
        client.post("/command", json={"command": "test"})
        
        # Verify the correct URL was used
        mock_streamable_client.assert_called_once_with("http://test-server:8080/mcp")


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('src.spotify_llm_controller.client.parse_command_with_llm')
    @patch('src.spotify_llm_controller.client.execute_spotify_actions')
    @patch('src.spotify_llm_controller.client.streamablehttp_client')
    @patch('src.spotify_llm_controller.client.ClientSession')
    def test_session_initialization_error(self, mock_client_session, mock_streamable_client, 
                                         mock_execute_actions, mock_parse_command):
        """Test handling of session initialization error."""
        mock_parse_command.return_value = [{"tool_name": "SpotifySearch", "params": {"query": "test"}}]
        
        mock_session = AsyncMock()
        mock_session.initialize.side_effect = Exception("Session init failed")
        mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_streamable_client.return_value.__aenter__ = AsyncMock(return_value=(None, None, None))
        mock_streamable_client.return_value.__aexit__ = AsyncMock(return_value=None)
        
        response = self.client.post("/command", json={"command": "play test song"})
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Session init failed" in data["error"]
    
    @patch('src.spotify_llm_controller.client.parse_command_with_llm')
    @patch('src.spotify_llm_controller.client.execute_spotify_actions')
    @patch('src.spotify_llm_controller.client.streamablehttp_client')
    @patch('src.spotify_llm_controller.client.ClientSession')
    def test_action_execution_error(self, mock_client_session, mock_streamable_client, 
                                   mock_execute_actions, mock_parse_command):
        """Test handling of action execution error."""
        mock_parse_command.return_value = [{"tool_name": "SpotifySearch", "params": {"query": "test"}}]
        mock_execute_actions.side_effect = Exception("Action execution failed")
        
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mock_streamable_client.return_value.__aenter__ = AsyncMock(return_value=(None, None, None))
        mock_streamable_client.return_value.__aexit__ = AsyncMock(return_value=None)
        
        response = self.client.post("/command", json={"command": "play test song"})
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Action execution failed" in data["error"]