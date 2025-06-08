# Spotify LLM Controller

A natural language interface for controlling Spotify through the MCP (Model Context Protocol) server.

## Features

- Natural language command parsing using OpenAI's LLM
- Spotify playback control (play, pause, skip)
- Search for tracks and albums
- Queue management
- Playlist management
- Robust error handling and retry logic

## Architecture

The client consists of several modules located in `src/spotify_llm_controller/`:

- `client.py`: Main FastAPI application with API endpoints
- `config.py`: Configuration settings and LLM prompts
- `openai_helper.py`: OpenAI API integration with version compatibility
- `spotify_actions.py`: Spotify action execution logic

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key
- Running Spotify MCP server

### Environment Variables

Create a `.env` file in the project root with the following variables (see `.env.example` for reference):

```
# Server configuration
MCP_SERVER_URL=http://127.0.0.1:8080
MCP_CLIENT_PORT=8090

# OpenAI configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_MAX_TOKENS=150

# Retry configuration
MAX_RETRIES=3
RETRY_DELAY=1
```

### Installation

1. Install dependencies:

```bash
pip install -e .
```

2. Run the client:

```bash
python -m src.spotify_llm_controller.client
```

Or using uvicorn directly:

```bash
uvicorn src.spotify_llm_controller.client:app --host 0.0.0.0 --port 8090
```

## API Endpoints

- `POST /command`: Send a natural language command to control Spotify
  - Request body: `{"command": "play bohemian rhapsody"}`
  - Response: Result of executing the command

- `GET /health`: Health check endpoint
  - Response: Service status information

- `GET /`: Root endpoint with API information
  - Response: Information about available endpoints

## Example Commands

- "Play Bohemian Rhapsody"
- "Queue up the new Taylor Swift album"
- "Pause the music"
- "Skip to the next track"
- "Play some jazz music"
- "Add this song to my favorites playlist"

## Development

### Adding New Features

To add support for new Spotify features:

1. Update the LLM prompt in `src/spotify_llm_controller/config.py` to include the new actions
2. Add handler functions in `src/spotify_llm_controller/spotify_actions.py`
3. Update tests to cover the new functionality

### Testing

Run tests with pytest:

```bash
pytest
```
