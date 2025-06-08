"""
Unit tests for the openai_helper module.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.spotify_llm_controller.openai_helper import OpenAIClient, parse_llm_response


class TestOpenAIClient:
    """Test the OpenAI client functionality."""
    
    def test_init_with_api_key(self):
        """Test initialization with provided API key."""
        client = OpenAIClient(api_key="test-key")
        assert client.api_key == "test-key"
    
    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch('src.spotify_llm_controller.openai_helper.OPENAI_API_KEY', None):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIClient()
    
    def test_init_with_config_api_key(self):
        """Test initialization using API key from config."""
        with patch('src.spotify_llm_controller.openai_helper.OPENAI_API_KEY', 'config-key'):
            client = OpenAIClient()
            assert client.api_key == "config-key"
    
    @patch('src.spotify_llm_controller.openai_helper.version')
    def test_version_detection(self, mock_version):
        """Test OpenAI SDK version detection."""
        mock_version.return_value = "1.5.0"
        client = OpenAIClient(api_key="test-key")
        assert client.openai_version == "1.5.0"
    
    @patch('src.spotify_llm_controller.openai_helper.version')
    def test_version_detection_failure(self, mock_version, caplog):
        """Test handling of version detection failure."""
        mock_version.side_effect = Exception("Version not found")
        client = OpenAIClient(api_key="test-key")
        assert client.openai_version == "unknown"
        assert "Could not determine OpenAI version" in caplog.text
    
    @patch('src.spotify_llm_controller.openai_helper.openai')
    def test_create_completion_legacy_sdk(self, mock_openai):
        """Test completion creation with legacy OpenAI SDK."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "  test response  "
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        client = OpenAIClient(api_key="test-key")
        client.openai_version = "0.28.0"  # Legacy version
        
        result = client.create_completion("test prompt")
        
        assert result == "test response"
        mock_openai.ChatCompletion.create.assert_called_once()
        assert mock_openai.api_key == "test-key"
    
    @patch('src.spotify_llm_controller.openai_helper.openai')
    def test_create_completion_new_sdk(self, mock_openai):
        """Test completion creation with new OpenAI SDK."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "  test response  "
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        
        client = OpenAIClient(api_key="test-key")
        client.openai_version = "1.5.0"  # New version
        
        result = client.create_completion("test prompt")
        
        assert result == "test response"
        mock_openai.OpenAI.assert_called_once_with(api_key="test-key")
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('src.spotify_llm_controller.openai_helper.openai')
    def test_create_completion_with_custom_parameters(self, mock_openai):
        """Test completion creation with custom parameters."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "response"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        
        client = OpenAIClient(api_key="test-key")
        client.openai_version = "1.5.0"
        
        result = client.create_completion(
            prompt="custom prompt",
            model="gpt-4",
            max_tokens=100,
            system_message="custom system"
        )
        
        # Verify the call was made with correct parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-4"
        assert call_args[1]['max_tokens'] == 100
        assert call_args[1]['messages'][0]['content'] == "custom system"
        assert call_args[1]['messages'][1]['content'] == "custom prompt"
    
    @patch('src.spotify_llm_controller.openai_helper.openai')
    def test_create_completion_api_error(self, mock_openai):
        """Test handling of OpenAI API errors."""
        mock_openai.OpenAI.side_effect = Exception("API Error")
        
        client = OpenAIClient(api_key="test-key")
        client.openai_version = "1.5.0"
        
        with pytest.raises(Exception, match="API Error"):
            client.create_completion("test prompt")


class TestParseLLMResponse:
    """Test the LLM response parsing functionality."""
    
    def test_parse_valid_json_list(self):
        """Test parsing a valid JSON list response."""
        response = '[{"tool_name": "SpotifySearch", "params": {"query": "test"}}]'
        result = parse_llm_response(response)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["tool_name"] == "SpotifySearch"
        assert result[0]["params"]["query"] == "test"
    
    def test_parse_valid_json_single_object(self):
        """Test parsing a single JSON object (backward compatibility)."""
        response = '{"tool_name": "SpotifyPlayback", "params": {"action": "pause"}}'
        result = parse_llm_response(response)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["tool_name"] == "SpotifyPlayback"
        assert result[0]["params"]["action"] == "pause"
    
    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON embedded in explanatory text."""
        response = '''Here's the action you requested:
        [{"tool_name": "SpotifySearch", "params": {"query": "test song"}}]
        This will search for the song.'''
        
        result = parse_llm_response(response)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["tool_name"] == "SpotifySearch"
    
    def test_parse_empty_response(self):
        """Test handling of empty response."""
        result = parse_llm_response("")
        assert "error" in result
        assert "empty response" in result["error"]
    
    def test_parse_invalid_json(self):
        """Test handling of invalid JSON."""
        result = parse_llm_response("This is not JSON at all")
        assert "error" in result
        assert "not valid JSON" in result["error"]
    
    def test_parse_malformed_json_in_text(self):
        """Test handling of malformed JSON within text."""
        response = 'Here is the action: [{"tool_name": "Test", "params":}] - malformed'
        result = parse_llm_response(response)
        assert "error" in result
        assert "Failed to parse" in result["error"]
    
    def test_validate_action_structure_missing_tool_name(self):
        """Test validation of action structure - missing tool_name."""
        response = '[{"params": {"query": "test"}}]'
        
        with pytest.raises(ValueError, match="Action missing tool_name"):
            parse_llm_response(response)
    
    def test_validate_action_structure_missing_params(self):
        """Test validation of action structure - missing params."""
        response = '[{"tool_name": "SpotifySearch"}]'
        
        with pytest.raises(ValueError, match="Action missing params"):
            parse_llm_response(response)
    
    def test_validate_action_structure_non_dict_action(self):
        """Test validation of action structure - non-dictionary action."""
        response = '["not a dictionary"]'
        
        with pytest.raises(ValueError, match="Each action must be a dictionary"):
            parse_llm_response(response)
    
    def test_parse_multiple_actions(self):
        """Test parsing multiple actions in a list."""
        response = '''[
            {"tool_name": "SpotifySearch", "params": {"query": "test"}},
            {"tool_name": "SpotifyPlayback", "params": {"action": "start"}}
        ]'''
        
        result = parse_llm_response(response)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["tool_name"] == "SpotifySearch"
        assert result[1]["tool_name"] == "SpotifyPlayback"
    
    def test_parse_complex_params(self):
        """Test parsing actions with complex parameter structures."""
        response = '''[{
            "tool_name": "SpotifyPlaylist",
            "params": {
                "action": "add_tracks",
                "playlist_id": "123",
                "track_ids": ["track1", "track2"],
                "metadata": {"source": "llm"}
            }
        }]'''
        
        result = parse_llm_response(response)
        
        assert len(result) == 1
        action = result[0]
        assert action["tool_name"] == "SpotifyPlaylist"
        assert action["params"]["action"] == "add_tracks"
        assert action["params"]["track_ids"] == ["track1", "track2"]
        assert action["params"]["metadata"]["source"] == "llm"