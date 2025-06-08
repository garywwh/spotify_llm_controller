"""
Unit tests for the spotify_actions module.
"""

import json
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.spotify_llm_controller.spotify_actions import (
    call_tool_with_retry,
    handle_search_action,
    handle_get_info_action,
    handle_playback_action,
    handle_queue_action,
    handle_playlist_action,
    execute_single_action,
    format_final_response,
    execute_spotify_actions
)


class TestCallToolWithRetry:
    """Test the retry mechanism for tool calls."""
    
    @pytest.mark.asyncio
    async def test_successful_call_first_attempt(self):
        """Test successful tool call on first attempt."""
        session = Mock()
        session.call_tool = AsyncMock(return_value="success")
        
        result = await call_tool_with_retry(session, "TestTool", {"param": "value"})
        
        assert result == "success"
        session.call_tool.assert_called_once_with("TestTool", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry mechanism when tool call fails."""
        session = Mock()
        session.call_tool = AsyncMock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            "success"
        ])
        
        with patch('src.spotify_llm_controller.spotify_actions.asyncio.sleep', new_callable=AsyncMock):
            result = await call_tool_with_retry(session, "TestTool", {"param": "value"}, max_retries=3)
        
        assert result == "success"
        assert session.call_tool.call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that exception is raised when max retries are exceeded."""
        session = Mock()
        session.call_tool = AsyncMock(side_effect=Exception("Persistent failure"))
        
        with patch('src.spotify_llm_controller.spotify_actions.asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception, match="Persistent failure"):
                await call_tool_with_retry(session, "TestTool", {"param": "value"}, max_retries=2)
        
        assert session.call_tool.call_count == 2


class TestHandleSearchAction:
    """Test the search action handler."""
    
    @pytest.mark.asyncio
    async def test_successful_track_search(self):
        """Test successful track search."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text=json.dumps({
            "tracks": [{
                "id": "track123",
                "name": "Test Song",
                "artists": ["Test Artist"]
            }]
        }))]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_search_action(session, {"query": "test song", "qtype": "track"})
        
        assert result["action"] == "search"
        assert result["result"]["name"] == "Test Song"
        assert result["result"]["artist"] == "Test Artist"
        assert result["result"]["type"] == "track"
        assert result["result"]["uri"] == "spotify:track:track123"
    
    @pytest.mark.asyncio
    async def test_successful_album_search(self):
        """Test successful album search."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text=json.dumps({
            "albums": [{
                "id": "album123",
                "name": "Test Album",
                "artists": ["Test Artist"]
            }]
        }))]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_search_action(session, {"query": "test album", "qtype": "album"})
        
        assert result["action"] == "search"
        assert result["result"]["name"] == "Test Album"
        assert result["result"]["artist"] == "Test Artist"
        assert result["result"]["type"] == "album"
        assert result["result"]["uri"] == "spotify:album:album123"
    
    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """Test search with no results."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text=json.dumps({}))]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_search_action(session, {"query": "nonexistent"})
        
        assert "error" in result
        assert "No results found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_search_error_response(self):
        """Test search with error response."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = True
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_search_action(session, {"query": "test"})
        
        assert "error" in result
        assert "No search results found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_search_invalid_json(self):
        """Test search with invalid JSON response."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="invalid json")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_search_action(session, {"query": "test"})
        
        assert "error" in result
        assert "Failed to parse search results" in result["error"]


class TestHandleGetInfoAction:
    """Test the get info action handler."""
    
    @pytest.mark.asyncio
    async def test_get_info_with_search_context(self):
        """Test get info action using search result from context."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text=json.dumps({
            "tracks": [{"id": "track1", "name": "Track 1"}]
        }))]
        
        context = {"search_result": {"uri": "spotify:album:album123"}}
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_get_info_action(session, {}, context)
        
        assert result["action"] == "get_info"
        assert "tracks" in result["result"]
        assert context["album_tracks"] == [{"id": "track1", "name": "Track 1"}]
    
    @pytest.mark.asyncio
    async def test_get_info_with_explicit_uri(self):
        """Test get info action with explicit item_uri."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text=json.dumps({
            "tracks": [{"id": "track1", "name": "Track 1"}]
        }))]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_get_info_action(session, {"item_uri": "spotify:album:explicit"}, {})
        
        assert result["action"] == "get_info"
    
    @pytest.mark.asyncio
    async def test_get_info_error_response(self):
        """Test get info with error response."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = True
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_get_info_action(session, {}, {})
        
        assert "error" in result
        assert "Failed to get album tracks" in result["error"]


class TestHandlePlaybackAction:
    """Test the playback action handler."""
    
    @pytest.mark.asyncio
    async def test_playback_start_with_search_result(self):
        """Test starting playback with search result."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        context = {
            "search_result": {
                "uri": "spotify:track:track123",
                "name": "Test Song",
                "artist": "Test Artist",
                "type": "track"
            }
        }
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playback_action(session, {"action": "start"}, context)
        
        assert result["action"] == "playback"
        assert result["type"] == "start"
        assert "Playing track Test Song by Test Artist" in result["message"]
        assert result["item"]["name"] == "Test Song"
    
    @pytest.mark.asyncio
    async def test_playback_resume_without_uri(self):
        """Test resuming playback without URI."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playback_action(session, {"action": "start"}, {})
        
        assert result["action"] == "playback"
        assert result["type"] == "resume"
        assert result["message"] == "Resuming playback"
    
    @pytest.mark.asyncio
    async def test_playback_pause(self):
        """Test pausing playback."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playback_action(session, {"action": "pause"}, {})
        
        assert result["action"] == "playback"
        assert result["type"] == "pause"
        assert result["message"] == "Playback paused"
    
    @pytest.mark.asyncio
    async def test_playback_skip(self):
        """Test skipping track."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playback_action(session, {"action": "skip"}, {})
        
        assert result["action"] == "playback"
        assert result["type"] == "skip"
        assert result["message"] == "Skipped to next track"
    
    @pytest.mark.asyncio
    async def test_playback_error_response(self):
        """Test playback with error response."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = True
        mock_result.content = [Mock(text="Playback failed")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playback_action(session, {"action": "start"}, {})
        
        assert "error" in result
        assert "Playback action failed" in result["error"]


class TestHandleQueueAction:
    """Test the queue action handler."""
    
    @pytest.mark.asyncio
    async def test_queue_with_search_result(self):
        """Test adding to queue with search result."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        context = {
            "search_result": {
                "uri": "spotify:track:track123",
                "name": "Test Song",
                "artist": "Test Artist",
                "type": "track"
            }
        }
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_queue_action(session, {"action": "add"}, context)
        
        assert result["action"] == "queue"
        assert "Added track Test Song by Test Artist to queue" in result["message"]
        assert result["item"]["name"] == "Test Song"
    
    @pytest.mark.asyncio
    async def test_queue_with_explicit_track_id(self):
        """Test adding to queue with explicit track ID."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_queue_action(session, {"action": "add", "track_id": "track123"}, {})
        
        assert result["action"] == "queue"
        assert result["message"] == "Added item to queue"
    
    @pytest.mark.asyncio
    async def test_queue_without_track_id_or_search_result(self):
        """Test queue action without track ID or search result."""
        session = Mock()
        
        result = await handle_queue_action(session, {"action": "add"}, {})
        
        assert "error" in result
        assert "No track ID available for queue" in result["error"]
    
    @pytest.mark.asyncio
    async def test_queue_error_response(self):
        """Test queue with error response."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = True
        mock_result.content = [Mock(text="Queue failed")]
        
        context = {"search_result": {"uri": "spotify:track:track123", "name": "Test", "artist": "Test", "type": "track"}}
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_queue_action(session, {"action": "add"}, context)
        
        assert "error" in result
        assert "Failed to add to queue" in result["error"]


class TestHandlePlaylistAction:
    """Test the playlist action handler."""
    
    @pytest.mark.asyncio
    async def test_playlist_get(self):
        """Test getting user playlists."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text=json.dumps([
            {"id": "playlist1", "name": "My Playlist 1"},
            {"id": "playlist2", "name": "My Playlist 2"}
        ]))]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playlist_action(session, {"action": "get"}, {})
        
        assert result["action"] == "playlist_get"
        assert "Found 2 playlists" in result["message"]
        assert len(result["playlists"]) == 2
    
    @pytest.mark.asyncio
    async def test_playlist_get_tracks(self):
        """Test getting tracks from a playlist."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text=json.dumps([
            {"id": "track1", "name": "Track 1"},
            {"id": "track2", "name": "Track 2"}
        ]))]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playlist_action(session, {"action": "get_tracks", "playlist_id": "playlist123"}, {})
        
        assert result["action"] == "playlist_get_tracks"
        assert "Found 2 tracks in playlist" in result["message"]
        assert len(result["tracks"]) == 2
    
    @pytest.mark.asyncio
    async def test_playlist_get_tracks_missing_id(self):
        """Test getting tracks without playlist ID."""
        session = Mock()
        
        result = await handle_playlist_action(session, {"action": "get_tracks"}, {})
        
        assert "error" in result
        assert "playlist_id is required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_playlist_add_tracks(self):
        """Test adding tracks to playlist."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playlist_action(session, {
                "action": "add_tracks",
                "playlist_id": "playlist123",
                "track_ids": ["track1", "track2"]
            }, {})
        
        assert result["action"] == "playlist_add_tracks"
        assert "Tracks added to playlist" in result["message"]
    
    @pytest.mark.asyncio
    async def test_playlist_remove_tracks(self):
        """Test removing tracks from playlist."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playlist_action(session, {
                "action": "remove_tracks",
                "playlist_id": "playlist123",
                "track_ids": ["track1", "track2"]
            }, {})
        
        assert result["action"] == "playlist_remove_tracks"
        assert "Tracks removed from playlist" in result["message"]
    
    @pytest.mark.asyncio
    async def test_playlist_change_details(self):
        """Test changing playlist details."""
        session = Mock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="success")]
        
        with patch('src.spotify_llm_controller.spotify_actions.call_tool_with_retry', 
                   new_callable=AsyncMock, return_value=mock_result):
            result = await handle_playlist_action(session, {
                "action": "change_details",
                "playlist_id": "playlist123",
                "name": "New Name"
            }, {})
        
        assert result["action"] == "playlist_change_details"
        assert "Playlist details updated" in result["message"]
    
    @pytest.mark.asyncio
    async def test_playlist_unknown_action(self):
        """Test unknown playlist action."""
        session = Mock()
        
        result = await handle_playlist_action(session, {"action": "unknown"}, {})
        
        assert "error" in result
        assert "Unknown playlist action: unknown" in result["error"]


class TestExecuteSingleAction:
    """Test the single action execution dispatcher."""
    
    @pytest.mark.asyncio
    async def test_execute_search_action(self):
        """Test executing a search action."""
        session = Mock()
        
        with patch('src.spotify_llm_controller.spotify_actions.handle_search_action', 
                   new_callable=AsyncMock, return_value={"action": "search"}) as mock_search:
            result = await execute_single_action(session, "SpotifySearch", {"query": "test"}, {})
        
        assert result["action"] == "search"
        mock_search.assert_called_once_with(session, {"query": "test"})
    
    @pytest.mark.asyncio
    async def test_execute_unsupported_action(self):
        """Test executing an unsupported action."""
        session = Mock()
        
        result = await execute_single_action(session, "UnsupportedTool", {}, {})
        
        assert "error" in result
        assert "Unsupported action: UnsupportedTool" in result["error"]


class TestFormatFinalResponse:
    """Test the final response formatting."""
    
    def test_format_all_successful(self):
        """Test formatting when all actions are successful."""
        results = [
            {"action": "search", "message": "Found song"},
            {"action": "playback", "message": "Playing song"}
        ]
        
        response = format_final_response(results)
        
        assert response["message"] == "Playing song"
        assert response["details"]["action"] == "playback"
    
    def test_format_all_failed(self):
        """Test formatting when all actions failed."""
        results = [
            {"error": "Search failed"},
            {"error": "Playback failed"}
        ]
        
        response = format_final_response(results)
        
        assert response["error"] == "Playback failed"
    
    def test_format_partial_success(self):
        """Test formatting when some actions succeeded and some failed."""
        results = [
            {"action": "search", "message": "Found song"},
            {"error": "Playback failed"}
        ]
        
        response = format_final_response(results)
        
        assert "Some actions failed" in response["message"]
        assert "success" in response["details"]
        assert "errors" in response["details"]


class TestExecuteSpotifyActions:
    """Test the main action execution function."""
    
    @pytest.mark.asyncio
    async def test_execute_successful_sequence(self):
        """Test executing a successful sequence of actions."""
        session = Mock()
        actions = [
            {"tool_name": "SpotifySearch", "params": {"query": "test song"}},
            {"tool_name": "SpotifyPlayback", "params": {"action": "start"}}
        ]
        
        search_result = {"action": "search", "result": {"uri": "spotify:track:123", "name": "Test"}}
        playback_result = {"action": "playback", "message": "Playing song"}
        
        with patch('src.spotify_llm_controller.spotify_actions.execute_single_action', 
                   new_callable=AsyncMock, side_effect=[search_result, playback_result]):
            result = await execute_spotify_actions(session, actions)
        
        assert "message" in result
        assert "details" in result
    
    @pytest.mark.asyncio
    async def test_execute_search_failure_stops_execution(self):
        """Test that search failure stops execution of subsequent actions."""
        session = Mock()
        actions = [
            {"tool_name": "SpotifySearch", "params": {"query": "nonexistent"}},
            {"tool_name": "SpotifyPlayback", "params": {"action": "start"}}
        ]
        
        search_result = {"error": "No results found"}
        
        with patch('src.spotify_llm_controller.spotify_actions.execute_single_action', 
                   new_callable=AsyncMock, return_value=search_result) as mock_execute:
            result = await execute_spotify_actions(session, actions)
        
        assert result["error"] == "No results found"
        # Should only call execute_single_action once (for the failed search)
        assert mock_execute.call_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_context_propagation(self):
        """Test that context is properly propagated between actions."""
        session = Mock()
        actions = [
            {"tool_name": "SpotifySearch", "params": {"query": "test"}},
            {"tool_name": "SpotifyPlayback", "params": {"action": "start"}}
        ]
        
        search_result = {"action": "search", "result": {"uri": "spotify:track:123", "name": "Test"}}
        playback_result = {"action": "playback", "message": "Playing"}
        
        with patch('src.spotify_llm_controller.spotify_actions.execute_single_action', 
                   new_callable=AsyncMock, side_effect=[search_result, playback_result]) as mock_execute:
            await execute_spotify_actions(session, actions)
        
        # Check that context was passed to the second action
        second_call_context = mock_execute.call_args_list[1][0][3]  # Fourth argument is context
        assert second_call_context["search_result"]["uri"] == "spotify:track:123"
    
    @pytest.mark.asyncio
    async def test_execute_exception_handling(self):
        """Test exception handling during action execution."""
        session = Mock()
        actions = [{"tool_name": "SpotifySearch", "params": {"query": "test"}}]
        
        with patch('src.spotify_llm_controller.spotify_actions.execute_single_action', 
                   new_callable=AsyncMock, side_effect=Exception("Unexpected error")):
            result = await execute_spotify_actions(session, actions)
        
        assert "error" in result
        assert "Unexpected error" in result["error"]