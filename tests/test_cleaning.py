import pytest
from src.__main__ import cleanup_output


def test_cleanup_windows_paths():
    """Ensure Windows paths are cleaned and backslashes preserved."""
    data = {"parameters": {"path": "C:\\\\Users\\\\john\\\\config.ini",
                           "encoding": "latin-1"}}
    prompt = "Read C:\\Users\\john\\config.ini with latin-1 encoding"

    result = cleanup_output(data, prompt)

    assert result["parameters"]["path"] == "C:\\Users\\john\\config.ini"
    assert result["parameters"]["encoding"] == "latin-1"


def test_cleanup_sql_query():
    """Ensure semantic noise appended to SQL queries is trimmed."""
    data = {"parameters": {"query": "'SELECT * FROM users' on the database"}}
    prompt = "Execute SQL query 'SELECT * FROM users'"

    result = cleanup_output(data, prompt)

    assert result["parameters"]["query"] == "SELECT * FROM users"


def test_cleanup_database_name():
    """Ensure repeated word 'database' is removed from the parameter."""
    data = {"parameters": {"database": "production database"}}
    prompt = "on the production database"

    result = cleanup_output(data, prompt)

    assert result["parameters"]["database"] == "production"


def test_empty_parameters():
    """Ensure code does not break when parameters JSON is empty."""
    data = {}
    prompt = "Do something"

    result = cleanup_output(data, prompt)

    assert result == {}
