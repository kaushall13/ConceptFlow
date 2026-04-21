"""utils/errors.py - Custom exception hierarchy for pipeline"""

from typing import Optional


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""

    def __init__(self, message: str, stage: str = "", recoverable: bool = True,
                 context: Optional[dict] = None):
        self.message = message
        self.stage = stage
        self.recoverable = recoverable
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        stage_info = f"[{self.stage}] " if self.stage else ""
        return f"{stage_info}{self.message}"


class ConfigurationError(PipelineError):
    """Exception for configuration issues."""

    def __init__(self, message: str, details: Optional[dict] = None):
        context = {"details": details} if details else {}
        super().__init__(message, stage="Configuration", recoverable=False, context=context)


class APIError(PipelineError):
    """Exception for API-related errors."""

    def __init__(self, message: str, api_name: str = "", status_code: Optional[int] = None):
        context = {"api_name": api_name}
        if status_code:
            context["status_code"] = status_code
        super().__init__(message, stage=f"{api_name} API", recoverable=True, context=context)


class ValidationError(PipelineError):
    """Exception for validation errors."""

    def __init__(self, message: str, field: str = "", value: any = None):
        context = {"field": field, "value": str(value)}
        super().__init__(message, stage="Validation", recoverable=False, context=context)


class StateError(PipelineError):
    """Exception for state management issues."""

    def __init__(self, message: str, state_file: str = ""):
        context = {"state_file": state_file}
        super().__init__(message, stage="State Management", recoverable=False, context=context)


class FileError(PipelineError):
    """Exception for file-related errors."""

    def __init__(self, message: str, file_path: str = ""):
        context = {"file_path": file_path}
        super().__init__(message, stage=f"File: {file_path}", recoverable=False, context=context)
