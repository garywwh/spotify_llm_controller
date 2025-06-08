"""
Spotify MCP Client

This module provides a FastAPI-based client for the Spotify MCP server.
It uses natural language processing to interpret user commands and
translates them into appropriate API calls to control Spotify.
"""

import asyncio
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# Import custom modules
from .config import MCP_SERVER_URL, MCP_CLIENT_PORT
from .openai_helper import OpenAIClient, parse_llm_response
from src.spotify_llm_controller.spotify_actions import execute_spotify_actions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Spotify MCP Client",
    description="Natural language interface for controlling Spotify",
    version="1.0.0"
)

class CommandRequest(BaseModel):
    command: str  # Natural language command from Telegram

def parse_command_with_llm(command: str) -> dict:
    """
    Use LLM to parse natural language commands into Spotify tool calls.
    
    Args:
        command: Natural language command from the user
        
    Returns:
        Parsed actions or error information
    """
    try:
        # Import here to avoid circular imports
        from .config import SPOTIFY_COMMAND_PROMPT
        
        # Format the prompt with the user's command
        prompt = SPOTIFY_COMMAND_PROMPT.format(command=command)
        
        # Initialize OpenAI client and get completion
        openai_client = OpenAIClient()
        content = openai_client.create_completion(prompt)
        
        logger.info(f"LLM response content: {content}")
        
        # Parse and validate the LLM response
        return parse_llm_response(content)
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        return {"error": f"Failed to parse command: {str(e)}"}

@app.post("/command")
async def handle_command(request: CommandRequest):
    """
    Handle natural language commands from the Telegram bot.
    
    Args:
        request: The command request containing the natural language command
        
    Returns:
        The result of executing the command
    """
    logger.info(f"Received command: {request.command}")

    # Parse the command using LLM
    actions = parse_command_with_llm(request.command)
    if "error" in actions:
        return actions

    # Execute the actions through MCP server
    mcp_endpoint = f"{MCP_SERVER_URL}/mcp"
    try:
        async with streamablehttp_client(mcp_endpoint) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await execute_spotify_actions(session, actions)
                logger.info(f"Action result: {result}")
                return result
    except Exception as e:
        logger.exception(f"Failed to execute command: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    
    Returns:
        Status information about the service
    """
    return {
        "status": "ok",
        "service": "spotify-mcp-client",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """
    Root endpoint with basic information about the API.
    
    Returns:
        Information about the API and available endpoints
    """
    return {
        "name": "Spotify MCP Client",
        "description": "Natural language interface for controlling Spotify",
        "endpoints": {
            "/command": "POST - Send a natural language command to control Spotify",
            "/health": "GET - Check the health of the service"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Print startup message
    print("\nSpotify MCP Client")
    print("=================")
    print(f"Server URL: {MCP_SERVER_URL}")
    print(f"Client Port: {MCP_CLIENT_PORT}")
    print(f"API Documentation: http://localhost:{MCP_CLIENT_PORT}/docs")
    print(f"Health Check: http://localhost:{MCP_CLIENT_PORT}/health")
    print(f"Send commands to: http://localhost:{MCP_CLIENT_PORT}/command")
    print("=================\n")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=MCP_CLIENT_PORT)
