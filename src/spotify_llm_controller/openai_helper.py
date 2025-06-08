"""
OpenAI integration helper for the Spotify MCP client.
"""

import json
import logging
import re
import openai
from typing import Dict, Any, Optional
from importlib.metadata import version
from .config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, SYSTEM_MESSAGE

logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Client for interacting with OpenAI API with version compatibility.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to value from config)
        
        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Determine OpenAI SDK version
        try:
            self.openai_version = version("openai")
        except Exception as e:
            logger.warning(f"Could not determine OpenAI version: {e}")
            self.openai_version = "unknown"
        logger.info(f"Using OpenAI SDK version: {self.openai_version}")
    
    def create_completion(self, prompt: str, model: str = OPENAI_MODEL, 
                         max_tokens: int = OPENAI_MAX_TOKENS, 
                         system_message: str = SYSTEM_MESSAGE) -> str:
        """
        Create a completion with the appropriate OpenAI SDK version.
        
        Args:
            prompt: The user prompt
            model: The OpenAI model to use
            max_tokens: Maximum tokens in the response
            system_message: System message for the LLM
            
        Returns:
            The completion text
            
        Raises:
            Exception: If the API call fails
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if self.openai_version.startswith("0."):
                # Legacy SDK
                openai.api_key = self.api_key
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            else:
                # New SDK
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

def parse_llm_response(content: str) -> Dict[str, Any]:
    """
    Parse the LLM response into a structured format.
    
    Args:
        content: The raw text response from the LLM
        
    Returns:
        Parsed actions or error information
        
    Raises:
        ValueError: If the response cannot be parsed
    """
    if not content:
        return {"error": "Received empty response from LLM"}
        
    try:
        # Try direct JSON parsing
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from text response (sometimes LLM adds explanatory text)
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except:
                return {"error": "Failed to parse LLM response as JSON"}
        else:
            return {"error": "LLM response is not valid JSON"}
    
    # Check if the response is an error object
    if isinstance(parsed, dict) and "error" in parsed and len(parsed) == 1:
        return parsed  # Return error response directly
    
    # Validate response structure
    if not isinstance(parsed, list):
        parsed = [parsed]  # Handle single action for backward compatibility
        
    for action in parsed:
        if not isinstance(action, dict):
            raise ValueError("Each action must be a dictionary")
        if "tool_name" not in action:
            raise ValueError("Action missing tool_name")
        if "params" not in action:
            raise ValueError("Action missing params")
        
    return parsed