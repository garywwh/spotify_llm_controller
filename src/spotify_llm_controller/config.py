"""
Configuration settings for the Spotify MCP client.

This module handles loading environment variables from .env files
and provides configuration values for the application.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)

# Determine the location of the .env file
# First check if .env exists in the current directory
env_path = Path('.env')
if not env_path.exists():
    # If not, check if it exists in the parent directory
    parent_env_path = Path('../.env')
    if parent_env_path.exists():
        env_path = parent_env_path
    else:
        # If not in parent directory, check in the mcp_client directory
        client_env_path = Path('mcp_client/.env')
        if client_env_path.exists():
            env_path = client_env_path

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)
logger.info(f"Loaded environment variables from {env_path.absolute() if env_path.exists() else 'environment'}")

# Server configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8080")
MCP_CLIENT_PORT = int(os.getenv("MCP_CLIENT_PORT", "8090"))

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "150"))

# Spotify command interpreter prompt
SPOTIFY_COMMAND_PROMPT = """You are a Spotify command interpreter. Your task is to interpret natural language commands into a series of actions to control Spotify playback.

Command: {command}

Available actions:
1. Search for tracks or albums:
{{
    "tool_name": "SpotifySearch",
    "params": {{
        "query": "<search term>",
        "qtype": "track,album",  # Use "track" for songs, "album" for albums, or "track,album" for both
        "limit": 1
    }}
}}

2. Control playback:
{{
    "tool_name": "SpotifyPlayback",
    "params": {{
        "action": "<start|pause|skip>",
        "spotify_uri": "<spotify uri for start action>"
    }}
}}

3. Manage queue:
{{
    "tool_name": "SpotifyQueue",
    "params": {{
        "action": "add",
        "track_id": "<spotify track id or uri>"  # For tracks or albums
    }}
}}

4. Get item info:
{{
    "tool_name": "SpotifyGetInfo",
    "params": {{
        "item_uri": "<spotify uri>"  # For getting album tracks
    }}
}}

5. Manage playlists:
{{
    "tool_name": "SpotifyPlaylist",
    "params": {{
        "action": "<get|get_tracks|add_tracks|remove_tracks|change_details>",
        "playlist_id": "<playlist id>",
        "track_ids": ["<track id>", "<track id>"],  # For add_tracks/remove_tracks
        "name": "<new name>",  # For change_details
        "description": "<new description>"  # For change_details
    }}
}}

For play/queue commands:

Playing a song:
1. First return a search action with qtype="track" to find the song
2. Then return a playback action to start playing it

Queueing a song:
1. First return a search action with qtype="track" to find the song
2. Then return a queue action with "action": "add" and let the spotify_uri be filled from search

Playing an album:
1. First return a search action with qtype="album" to find the album
2. Return a playback action with the album URI for playing

Queueing an album:
1. First return a search action with qtype="album" to find the album
2. Return a queue action with the album URI for queueing

For pause/resume:
- Return a playback action with "pause" or "start"
- For just "play" without specifying content, return a start action without spotify_uri to resume current playback

For skip/next:
- Return a playback action with "skip"

Examples:

For playing a song:
[
    {{"tool_name": "SpotifySearch", "params": {{"query": "song name", "qtype": "track", "limit": 1}}}},
    {{"tool_name": "SpotifyPlayback", "params": {{"action": "start"}}}}
]

For queueing an album:
[
    {{"tool_name": "SpotifySearch", "params": {{"query": "album name", "qtype": "album", "limit": 1}}}},
    {{"tool_name": "SpotifyGetInfo", "params": {{"item_uri": null}}}},
    {{"tool_name": "SpotifyQueue", "params": {{"action": "add", "spotify_uri": null}}}}
]

For resuming playback (just "play"):
[
    {{"tool_name": "SpotifyPlayback", "params": {{"action": "start"}}}}
]

The spotify_uri and item_uri in subsequent actions will be filled in with the result from the search."""

# System message for the LLM
SYSTEM_MESSAGE = "You are a Spotify command interpreter. For play commands, you MUST return both search and playback actions together. The search action finds the track/album, and the playback action starts playing it."

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds