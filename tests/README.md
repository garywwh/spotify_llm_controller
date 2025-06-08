# Spotify LLM Controller Tests

This directory contains comprehensive unit tests for the Spotify LLM Controller project.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and configuration
├── test_config.py           # Tests for configuration module
├── test_openai_helper.py    # Tests for OpenAI integration
├── test_spotify_actions.py  # Tests for Spotify action execution
├── test_client.py           # Tests for FastAPI client endpoints
└── README.md               # This file
```

## Test Categories

### Unit Tests
- **test_config.py**: Tests configuration loading, environment variable handling, and default values
- **test_openai_helper.py**: Tests OpenAI client initialization, completion creation, and response parsing
- **test_spotify_actions.py**: Tests individual Spotify action handlers and execution logic
- **test_client.py**: Tests FastAPI endpoints, request handling, and integration flows

### Test Coverage

The tests cover:

1. **Configuration Management**
   - Environment variable loading
   - Default value handling
   - Configuration validation
   - Prompt template validation

2. **OpenAI Integration**
   - Client initialization with different SDK versions
   - Completion creation with various parameters
   - Response parsing and validation
   - Error handling for API failures

3. **Spotify Actions**
   - Search functionality (tracks and albums)
   - Playback control (start, pause, skip)
   - Queue management
   - Playlist operations
   - Context propagation between actions
   - Retry mechanisms
   - Error handling

4. **FastAPI Client**
   - Endpoint functionality (/command, /health, /)
   - Request validation
   - Command parsing integration
   - MCP session management
   - Error handling and logging

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e ".[test]"
```

### Basic Test Execution

Run all tests:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run specific test file:
```bash
pytest tests/test_config.py
```

Run specific test class:
```bash
pytest tests/test_openai_helper.py::TestOpenAIClient
```

Run specific test method:
```bash
pytest tests/test_client.py::TestFastAPIEndpoints::test_health_endpoint
```

### Test Coverage

Generate coverage report:
```bash
pytest --cov=src/spotify_llm_controller --cov-report=html --cov-report=term-missing
```

View HTML coverage report:
```bash
open htmlcov/index.html
```

### Using Makefile

The project includes a Makefile for common tasks:

```bash
make test              # Run all tests
make test-verbose      # Run tests with verbose output
make test-coverage     # Run tests with coverage report
make test-unit         # Run only unit tests
```

## Test Fixtures

The `conftest.py` file provides shared fixtures:

- `mock_mcp_session`: Mock MCP client session
- `mock_openai_client`: Mock OpenAI client
- `sample_search_result`: Sample search result data
- `mock_mcp_result_success`: Mock successful MCP response
- `mock_mcp_result_error`: Mock error MCP response
- `sample_actions_sequence`: Sample action sequences for testing

## Mocking Strategy

The tests use extensive mocking to isolate units under test:

1. **External Dependencies**: OpenAI API, MCP server connections
2. **Async Operations**: All async calls are mocked with AsyncMock
3. **Network Calls**: HTTP clients and streamable connections
4. **Environment Variables**: Configuration values for testing

## Test Patterns

### Async Testing
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected_value
```

### Mocking External Services
```python
@patch('module.external_service')
def test_with_mock(mock_service):
    mock_service.return_value = "mocked_response"
    result = function_under_test()
    assert result == expected_result
```

### Testing Error Conditions
```python
def test_error_handling():
    with pytest.raises(ValueError, match="expected error message"):
        function_that_should_raise_error()
```

## Continuous Integration

The tests are designed to run in CI environments:

- No external dependencies required
- All network calls are mocked
- Environment variables are controlled
- Deterministic test execution

## Adding New Tests

When adding new functionality:

1. Create corresponding test methods
2. Use appropriate fixtures from `conftest.py`
3. Mock external dependencies
4. Test both success and error cases
5. Maintain test isolation
6. Update this README if needed

## Test Quality Guidelines

- **Isolation**: Each test should be independent
- **Clarity**: Test names should describe what is being tested
- **Coverage**: Test both happy path and error conditions
- **Maintainability**: Use fixtures and helpers to reduce duplication
- **Performance**: Tests should run quickly (< 1 second each)

## Debugging Tests

For debugging failing tests:

```bash
# Run with detailed output
pytest -v -s

# Run specific test with debugging
pytest tests/test_client.py::test_specific_function -v -s

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l
```