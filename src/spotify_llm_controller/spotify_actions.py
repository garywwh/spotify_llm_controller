"""
Spotify action execution module for the MCP client.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from mcp import ClientSession

from .config import MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger(__name__)

async def call_tool_with_retry(session: ClientSession, tool_name: str, params: Dict[str, Any], 
                              max_retries: int = MAX_RETRIES) -> Any:
    """
    Call a tool with retry logic for transient failures.
    
    Args:
        session: The MCP client session
        tool_name: Name of the tool to call
        params: Parameters for the tool
        max_retries: Maximum number of retry attempts
        
    Returns:
        The tool response
        
    Raises:
        Exception: If all retry attempts fail
    """
    retries = 0
    while retries < max_retries:
        try:
            result = await session.call_tool(tool_name, params)
            return result
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Retry {retries}/{max_retries} after error: {e}")
            # Exponential backoff
            await asyncio.sleep(RETRY_DELAY * (2 ** (retries - 1)))

async def handle_search_action(session: ClientSession, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a Spotify search action.
    
    Args:
        session: The MCP client session
        params: Search parameters
        
    Returns:
        Search result or error information
    """
    # Do the search
    result = await call_tool_with_retry(session, "SpotifySearch", params)
    logger.info(f"Search result: {result}")
    
    if not result or result.isError:
        error_msg = "No search results found"
        logger.error(error_msg)
        return {"error": error_msg}

    try:
        # Extract the text content from the first message
        result_text = next((msg.text for msg in result.content if hasattr(msg, 'text')), '{}')
        result_data = json.loads(result_text)
        logger.info(f"Parsed search result: {result_data}")
        
        # Handle both track and album results
        if params.get("qtype") == "album" and result_data.get("albums"):
            album = result_data["albums"][0]
            logger.debug(f"Album data: {album}")
            
            # Handle different artist formats (single string or list)
            if "artists" in album and isinstance(album["artists"], list):
                artist = album["artists"][0]  # Use the first artist
                logger.debug(f"Using first artist from list: {artist}")
            else:
                artist = album.get("artist", "Unknown Artist")
                logger.debug(f"Using artist field: {artist}")
                
            search_result = {
                "name": album["name"],
                "uri": f"spotify:album:{album['id']}",
                "id": album["id"],
                "artist": artist,
                "type": "album"
            }
            logger.info(f"Parsed album result: {search_result}")
        elif result_data.get("tracks"):
            track = result_data["tracks"][0]
            logger.debug(f"Track data: {track}")
            
            # Handle different artist formats (single string or list)
            if "artists" in track and isinstance(track["artists"], list):
                artist = track["artists"][0]  # Use the first artist
                logger.debug(f"Using first artist from list: {artist}")
            else:
                artist = track.get("artist", "Unknown Artist")
                logger.debug(f"Using artist field: {artist}")
                
            search_result = {
                "name": track["name"],
                "uri": f"spotify:track:{track['id']}",
                "id": track["id"],
                "artist": artist,
                "type": "track"
            }
            logger.info(f"Parsed track result: {search_result}")
        else:
            error_msg = f"No results found for query: {params.get('query')}"
            logger.error(error_msg)
            return {"error": error_msg}

        return {
            "action": "search",
            "result": search_result
        }
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"Failed to parse search results: {e}")
        return {"error": "Failed to parse search results"}

async def handle_get_info_action(session: ClientSession, params: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a Spotify get info action.
    
    Args:
        session: The MCP client session
        params: Get info parameters
        context: Execution context with previous results
        
    Returns:
        Album tracks or error information
    """
    # If we have a search result and no item_uri specified, use the search result
    search_result = context.get("search_result")
    if search_result and not params.get("item_uri"):
        params["item_uri"] = search_result["uri"]
    
    # Get album tracks
    result = await call_tool_with_retry(session, "SpotifyGetInfo", params)
    logger.info(f"Get info result: {result}")
    
    if result.isError:
        error_msg = "Failed to get album tracks"
        logger.error(error_msg)
        return {"error": error_msg}

    try:
        result_text = next((msg.text for msg in result.content if hasattr(msg, 'text')), '{}')
        album_data = json.loads(result_text)
        album_tracks = album_data.get("tracks", [])
        if not album_tracks:
            error_msg = "No tracks found in album"
            logger.error(error_msg)
            return {"error": error_msg}
        
        context["album_tracks"] = album_tracks
        return {
            "action": "get_info",
            "result": album_data
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse album tracks: {e}")
        return {"error": "Failed to parse album information"}

async def handle_playback_action(session: ClientSession, params: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a Spotify playback action.
    
    Args:
        session: The MCP client session
        params: Playback parameters
        context: Execution context with previous results
        
    Returns:
        Playback result or error information
    """
    search_result = context.get("search_result")
    
    # Validate parameters for start action
    if params.get("action") == "start":
        # If no URI is provided and no search result, this is a resume command
        if not params.get("spotify_uri") and not search_result:
            logger.info("No URI provided for start action - will resume current playback")
            # This is valid - the server will resume current playback

        # If this is a start action and we have a search result, use that URI
        if search_result and not params.get("spotify_uri"):
            # Ensure proper URI format
            if ":" not in search_result["uri"]:
                params["spotify_uri"] = f"spotify:track:{search_result['uri']}"
            else:
                params["spotify_uri"] = search_result["uri"]
            logger.info(f"Starting playback with URI: {params['spotify_uri']}")

    # Debug log the final params
    logger.info(f"Final playback parameters: {json.dumps(params, indent=2)}")
    
    # Execute the playback action
    result = await call_tool_with_retry(session, "SpotifyPlayback", params)
    logger.info(f"Raw playback result: {result}")
    if hasattr(result, 'content'):
        for msg in result.content:
            if hasattr(msg, 'text'):
                logger.info(f"Playback response content: {msg.text}")
    
    if result.isError:
        error_msg = f"Playback action failed: {params.get('action')}"
        logger.error(error_msg)
        # Include more detailed error info if available
        if hasattr(result, 'content'):
            try:
                error_content = next((msg.text for msg in result.content if hasattr(msg, 'text')), '')
                logger.error(f"Detailed error: {error_content}")
                return {"error": f"{error_msg}: {error_content}"}
            except Exception as e:
                logger.error(f"Error parsing failure response: {e}")
                return {"error": error_msg}
        return {"error": error_msg}
    
    # Format response based on action
    if params.get("action") == "start":
        if search_result:
            item_type = "album" if search_result["type"] == "album" else "track"
            msg = f"Playing {item_type} {search_result['name']} by {search_result['artist']}"
            
            action_result = {
                "action": "playback",
                "type": params.get("action"),
                "message": msg,
                "item": search_result
            }
        else:
            # This is a resume command
            action_result = {
                "action": "playback",
                "type": "resume",
                "message": "Resuming playback"
            }
    elif params.get("action") == "pause":
        action_result = {
            "action": "playback",
            "type": "pause",
            "message": "Playback paused"
        }
    elif params.get("action") == "skip":
        action_result = {
            "action": "playback",
            "type": "skip",
            "message": "Skipped to next track"
        }
    else:
        action_result = {
            "action": "playback",
            "type": params.get("action", "unknown"),
            "message": "Playback command executed"
        }
    
    return action_result

async def handle_queue_action(session: ClientSession, params: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a Spotify queue action.
    
    Args:
        session: The MCP client session
        params: Queue parameters
        context: Execution context with previous results
        
    Returns:
        Queue result or error information
    """
    search_result = context.get("search_result")
    
    # Always need either a track_id in params or a search_result
    if not params.get("track_id") and not search_result:
        error_msg = "No track ID available for queue"
        logger.error(error_msg)
        return {"error": error_msg}

    # If we have a search result, use its URI as track_id
    if search_result and not params.get("track_id"):
        params["track_id"] = search_result["uri"]  # Already properly formatted from search
        logger.info(f"Queueing with track_id: {params['track_id']}")

    # Debug log the final params
    logger.info(f"Final queue parameters: {json.dumps(params, indent=2)}")

    # Add to queue
    result = await call_tool_with_retry(session, "SpotifyQueue", params)
    logger.info(f"Raw queue result: {result}")

    # Log response content for debugging
    if hasattr(result, 'content'):
        for msg in result.content:
            if hasattr(msg, 'text'):
                logger.info(f"Queue response content: {msg.text}")
    
    if result.isError:
        error_msg = "Failed to add to queue"
        logger.error(error_msg)
        if hasattr(result, 'content'):
            try:
                error_content = next((msg.text for msg in result.content if hasattr(msg, 'text')), '')
                logger.error(f"Detailed error: {error_content}")
                return {"error": f"{error_msg}: {error_content}"}
            except Exception as e:
                logger.error(f"Error parsing failure response: {e}")
                return {"error": error_msg}
        return {"error": error_msg}

    if search_result:
        item_type = "album" if search_result["type"] == "album" else "track"
        msg = f"Added {item_type} {search_result['name']} by {search_result['artist']} to queue"
        
        return {
            "action": "queue",
            "message": msg,
            "item": search_result
        }
    else:
        return {
            "action": "queue",
            "message": "Added item to queue"
        }

async def handle_playlist_action(session: ClientSession, params: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a Spotify playlist action.
    
    Args:
        session: The MCP client session
        params: Playlist parameters
        context: Execution context with previous results
        
    Returns:
        Playlist result or error information
    """
    action_type = params.get("action")
    
    if action_type == "get":
        # Get user's playlists
        result = await call_tool_with_retry(session, "SpotifyPlaylist", params)
        if result.isError:
            return {"error": "Failed to get playlists"}
            
        try:
            result_text = next((msg.text for msg in result.content if hasattr(msg, 'text')), '{}')
            playlists = json.loads(result_text)
            return {
                "action": "playlist_get",
                "message": f"Found {len(playlists)} playlists",
                "playlists": playlists
            }
        except Exception as e:
            logger.error(f"Failed to parse playlists: {e}")
            return {"error": "Failed to parse playlists"}
            
    elif action_type == "get_tracks":
        # Get tracks from a playlist
        if not params.get("playlist_id"):
            return {"error": "playlist_id is required for get_tracks action"}
            
        result = await call_tool_with_retry(session, "SpotifyPlaylist", params)
        if result.isError:
            return {"error": "Failed to get playlist tracks"}
            
        try:
            result_text = next((msg.text for msg in result.content if hasattr(msg, 'text')), '{}')
            tracks = json.loads(result_text)
            return {
                "action": "playlist_get_tracks",
                "message": f"Found {len(tracks)} tracks in playlist",
                "tracks": tracks
            }
        except Exception as e:
            logger.error(f"Failed to parse playlist tracks: {e}")
            return {"error": "Failed to parse playlist tracks"}
            
    elif action_type in ["add_tracks", "remove_tracks"]:
        # Add or remove tracks from a playlist
        if not params.get("playlist_id") or not params.get("track_ids"):
            return {"error": f"playlist_id and track_ids are required for {action_type} action"}
            
        result = await call_tool_with_retry(session, "SpotifyPlaylist", params)
        if result.isError:
            return {"error": f"Failed to {action_type.replace('_', ' ')}"}
            
        action_name = "added to" if action_type == "add_tracks" else "removed from"
        return {
            "action": f"playlist_{action_type}",
            "message": f"Tracks {action_name} playlist"
        }
        
    elif action_type == "change_details":
        # Change playlist details
        if not params.get("playlist_id") or (not params.get("name") and not params.get("description")):
            return {"error": "playlist_id and at least one of name or description are required"}
            
        result = await call_tool_with_retry(session, "SpotifyPlaylist", params)
        if result.isError:
            return {"error": "Failed to change playlist details"}
            
        return {
            "action": "playlist_change_details",
            "message": "Playlist details updated"
        }
        
    else:
        return {"error": f"Unknown playlist action: {action_type}"}

async def execute_single_action(session: ClientSession, tool_name: str, params: Dict[str, Any], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single Spotify action.
    
    Args:
        session: The MCP client session
        tool_name: Name of the tool to call
        params: Parameters for the tool
        context: Execution context with previous results
        
    Returns:
        Action result or error information
    """
    if tool_name == "SpotifySearch":
        return await handle_search_action(session, params)
    elif tool_name == "SpotifyGetInfo":
        return await handle_get_info_action(session, params, context)
    elif tool_name == "SpotifyPlayback":
        return await handle_playback_action(session, params, context)
    elif tool_name == "SpotifyQueue":
        return await handle_queue_action(session, params, context)
    elif tool_name == "SpotifyPlaylist":
        return await handle_playlist_action(session, params, context)
    else:
        logger.error(f"Unsupported tool: {tool_name}")
        return {"error": f"Unsupported action: {tool_name}"}

def format_final_response(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format the final response from a sequence of action results.
    
    Args:
        results: List of action results
        
    Returns:
        Formatted response for the user
    """
    # Return the final user-facing message
    all_errors = [r for r in results if "error" in r]
    if all_errors:
        # If all actions failed, return the last error
        if len(all_errors) == len(results):
            return all_errors[-1]
        # If some actions succeeded, return both success and error info
        successful_actions = [r for r in results if "error" not in r]
        return {
            "message": successful_actions[-1].get("message", "Partial success") + "\nSome actions failed",
            "details": {
                "success": successful_actions[-1],
                "errors": all_errors
            }
        }
    
    return {
        "message": results[-1].get("message", "Command executed successfully"),
        "details": results[-1]
    }

async def execute_spotify_actions(session: ClientSession, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute a sequence of Spotify actions through the MCP server.
    
    This function processes a list of actions sequentially, maintaining context
    between actions (e.g., using search results in subsequent playback actions).
    
    Args:
        session: The MCP client session
        actions: List of action dictionaries, each containing:
            - tool_name: The MCP tool to call
            - params: Parameters for the tool
            
    Returns:
        Result of the action sequence, containing:
            - message: User-friendly message about the result
            - details: Detailed information about the executed actions
            - error: Error message if the action failed
            
    Raises:
        Exception: If there's an unhandled error during execution
    """
    try:
        results = []
        context = {"search_result": None, "album_tracks": None}
        
        for action in actions:
            tool_name = action["tool_name"]
            params = action["params"]
            logger.info(f"Executing action: {tool_name} with params: {params}")
            
            result = await execute_single_action(session, tool_name, params, context)
            results.append(result)
            
            # If this action failed and it's critical (like search), return immediately
            if "error" in result and tool_name == "SpotifySearch":
                logger.error(f"Critical action {tool_name} failed: {result['error']}")
                return result
            
            # Update context with results if needed
            if "error" not in result:
                if result.get("action") == "search":
                    context["search_result"] = result.get("result")
                    logger.info(f"Updated search context: {context['search_result']}")
                elif result.get("action") == "get_info":
                    context["album_info"] = result.get("result")
                    logger.info(f"Updated album info context: {context['album_info']}")
        
        return format_final_response(results)
    except Exception as e:
        logger.exception(f"Error executing Spotify action: {e}")
        return {"error": str(e)}