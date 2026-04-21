"""
Error Handling and Logging Utilities
"""

import sys
import traceback
from typing import Any, Callable, Optional
from functools import wraps


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(self, message: str, stage: str = "", recoverable: bool = True):
        self.message = message
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(message)


class ConfigurationError(PipelineError):
    """Exception for configuration issues."""

    def __init__(self, message: str):
        super().__init__(message, stage="Configuration", recoverable=False)


class APIError(PipelineError):
    """Exception for API-related errors."""

    def __init__(self, message: str, api_name: str = "", recoverable: bool = True):
        stage = f"{api_name} API" if api_name else "API"
        super().__init__(message, stage=stage, recoverable=recoverable)


class FileError(PipelineError):
    """Exception for file-related errors."""

    def __init__(self, message: str, file_path: str = ""):
        stage = f"File: {file_path}" if file_path else "File"
        super().__init__(message, stage=stage, recoverable=False)


def handle_pipeline_errors(func: Callable) -> Callable:
    """
    Decorator to handle pipeline errors consistently.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with error handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PipelineError as e:
            print_error(f"Pipeline Error [{e.stage}]: {e.message}")

            if not e.recoverable:
                print_error("This error cannot be recovered from. Please fix the issue and try again.")
                sys.exit(1)

            # Offer recovery options for recoverable errors
            return handle_recoverable_error(e)
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print_error(f"Unexpected Error: {str(e)}")
            print_error("Please report this issue with the following details:")
            traceback.print_exc()
            sys.exit(1)

    return wrapper


def print_error(message: str, prefix: str = "X") -> None:
    """
    Print error message with consistent formatting.

    Args:
        message: Error message to print
        prefix: Prefix symbol (default: X)
    """
    print(f"{prefix} {message}", file=sys.stderr)


def print_warning(message: str, prefix: str = "WARNING") -> None:
    """
    Print warning message with consistent formatting.

    Args:
        message: Warning message to print
        prefix: Prefix symbol (default: WARNING)
    """
    print(f"{prefix}: {message}", file=sys.stderr)


def print_success(message: str, prefix: str = "SUCCESS") -> None:
    """
    Print success message with consistent formatting.

    Args:
        message: Success message to print
        prefix: Prefix symbol (default: SUCCESS)
    """
    print(f"{prefix}: {message}")


def print_info(message: str, prefix: str = "INFO") -> None:
    """
    Print info message with consistent formatting.

    Args:
        message: Info message to print
        prefix: Prefix symbol (default: INFO)
    """
    print(f"{prefix}: {message}")


def handle_recoverable_error(error: PipelineError) -> Optional[Any]:
    """
    Handle recoverable errors by offering user choices.

    Args:
        error: The recoverable error

    Returns:
        User's choice or None
    """
    print()
    print("Recovery options:")
    print("  [R]etry - Attempt the operation again")
    print("  [S]kip - Skip this step and continue")
    print("  [A]bort - Stop the pipeline")

    while True:
        choice = input("\nYour choice [R/S/A]: ").strip().upper()

        if choice == 'R':
            return 'retry'
        elif choice == 'S':
            return 'skip'
        elif choice == 'A':
            print("Pipeline aborted.")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter R, S, or A.")


def validate_pdf_path(pdf_path: str) -> str:
    """
    Validate PDF file path and return normalized path.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Normalized PDF path

    Raises:
        FileError: If PDF file doesn't exist or is invalid
    """
    from pathlib import Path

    path = Path(pdf_path)

    if not path.exists():
        raise FileError(f"PDF file not found: {pdf_path}", pdf_path)

    if not path.is_file():
        raise FileError(f"Path is not a file: {pdf_path}", pdf_path)

    if path.suffix.lower() != '.pdf':
        raise FileError(f"File is not a PDF: {pdf_path}", pdf_path)

    return str(path.resolve())


def validate_config(config: dict) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ConfigurationError: If configuration is invalid
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
        raise ConfigurationError("Cerebras API key seems too short. Please check your configuration.")

    # Validate Ollama settings if provided
    ollama_host = config.get('ollama_host', '')
    if ollama_host and not ollama_host.startswith('http'):
        raise ConfigurationError("Ollama host must start with http:// or https://")


def log_stage_start(stage_name: str) -> None:
    """
    Log the start of a pipeline stage.

    Args:
        stage_name: Name of the stage
    """
    print()
    print("=" * 60)
    print(f"STAGE: {stage_name}")
    print("=" * 60)


def log_stage_complete(stage_name: str, details: str = "") -> None:
    """
    Log the completion of a pipeline stage.

    Args:
        stage_name: Name of the stage
        details: Optional details to log
    """
    print_success(f"{stage_name} complete{' - ' + details if details else ''}")


def log_progress(current: int, total: int, item_name: str = "items") -> None:
    """
    Log progress for multi-item operations.

    Args:
        current: Current item number
        total: Total number of items
        item_name: Name of items being processed
    """
    percentage = (current / total) * 100
    print(f"  Progress: {current}/{total} {item_name} ({percentage:.1f}%)")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_user_confirmation(message: str, default: bool = False) -> bool:
    """
    Get yes/no confirmation from user.

    Args:
        message: Confirmation message
        default: Default value if user just presses Enter

    Returns:
        True if user confirms, False otherwise
    """
    prompt = f"{message} [Y/n]: " if default else f"{message} [y/N]: "

    while True:
        choice = input(prompt).strip().lower()

        if not choice:
            return default

        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter Y or N.")