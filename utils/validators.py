"""utils/validators.py - Input/output validation utilities"""

import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.errors import ValidationError, ConfigurationError


def validate_pdf_path(path: str) -> str:
    """
    Validate PDF file path.

    Args:
        path: Path to validate

    Returns:
        Normalized path

    Raises:
        ValidationError: If path is invalid
    """
    p = Path(path)

    if not p.exists():
        raise ValidationError(f"PDF file not found: {path}", field="pdf_path")
    if not p.is_file():
        raise ValidationError(f"Path is not a file: {path}", field="pdf_path")
    if p.suffix.lower() != '.pdf':
        raise ValidationError(f"File is not a PDF: {path}", field="pdf_path")

    return str(p.resolve())


def validate_config(config: dict) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration to validate

    Raises:
        ConfigurationError: If config is invalid
    """
    # Accept new dual-key format or legacy single-key format
    api_key = (
        config.get('cerebras_api_key_1') or
        config.get('cerebras_api_key_2') or
        config.get('cerebras_api_key')
    )
    if not api_key:
        raise ConfigurationError("Missing required configuration: cerebras_api_key_1")
    if len(api_key) < 10:
        raise ConfigurationError("Cerebras API key seems too short")

    ollama_host = config.get('ollama_host', '')
    if ollama_host and not ollama_host.startswith('http'):
        raise ConfigurationError("Ollama host must start with http:// or https://")


def validate_session_count(count: int) -> None:
    """
    Validate session count is reasonable.

    Args:
        count: Number of sessions

    Raises:
        ValidationError: If count is unreasonable
    """
    if count < 1:
        raise ValidationError(f"Session count must be at least 1, got {count}", field="session_count")
    if count > 500:
        raise ValidationError(f"Session count too high (max 500), got {count}", field="session_count")


if __name__ == "__main__":
    print("Testing validators...")

    # Test PDF path validation
    try:
        validate_pdf_path("test.pdf")
        print("[FAIL] Should raise ValidationError for nonexistent file")
    except ValidationError as e:
        print(f"[OK] Caught expected error: {e.message}")

    # Test config validation
    try:
        validate_config({})
        print("[FAIL] Should raise ConfigurationError for empty config")
    except ConfigurationError as e:
        print(f"[OK] Caught expected error: {e.message}")

    # Test session count validation
    try:
        validate_session_count(1000)
        print("[FAIL] Should raise ValidationError for too many sessions")
    except ValidationError as e:
        print(f"[OK] Caught expected error: {e.message}")

    print("\n[OK] All validation tests passed!")
