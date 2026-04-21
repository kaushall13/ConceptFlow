"""
Utils Package - Common utilities for the application
"""

from .error_handler import (
    PipelineError,
    ConfigurationError,
    APIError,
    FileError,
    handle_pipeline_errors,
    print_error,
    print_warning,
    print_success,
    print_info,
    handle_recoverable_error,
    validate_pdf_path,
    validate_config,
    log_stage_start,
    log_stage_complete,
    log_progress,
    format_duration,
    get_user_confirmation
)

__all__ = [
    'PipelineError',
    'ConfigurationError',
    'APIError',
    'FileError',
    'handle_pipeline_errors',
    'print_error',
    'print_warning',
    'print_success',
    'print_info',
    'handle_recoverable_error',
    'validate_pdf_path',
    'validate_config',
    'log_stage_start',
    'log_stage_complete',
    'log_progress',
    'format_duration',
    'get_user_confirmation'
]