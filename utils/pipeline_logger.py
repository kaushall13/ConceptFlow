"""utils/pipeline_logger.py - Structured logging for pipeline"""

import sys
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

# Import standard library logging
import logging as stdlib_logging


class PipelineLogger:
    """Structured logger with timestamps, levels, and context."""

    def __init__(self, name: str, level: int = stdlib_logging.INFO):
        """
        Initialize logger.

        Args:
            name: Logger name (usually module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = stdlib_logging.getLogger(name)
        self.logger.setLevel(level)
        self.stage = ""

        # Console handler with formatting
        handler = stdlib_logging.StreamHandler(sys.stdout)
        formatter = stdlib_logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(name)s | %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    @contextmanager
    def stage_context(self, stage_name: str):
        """Context manager for logging within a pipeline stage."""
        old_stage = self.stage
        try:
            self.stage = stage_name
            yield self.logger
        finally:
            self.stage = old_stage

    def _format_message(self, message: str) -> str:
        """Format message with stage context if available."""
        if self.stage:
            return f"[{self.stage}] {message}"
        return message

    def info(self, message: str):
        """Log info message with optional context."""
        self.logger.info(self._format_message(message))

    def warning(self, message: str):
        """Log warning message with optional context."""
        self.logger.warning(self._format_message(message))

    def error(self, message: str):
        """Log error message with optional context."""
        self.logger.error(self._format_message(message))

    def debug(self, message: str):
        """Log debug message with optional context."""
        self.logger.debug(self._format_message(message))


# Convenience functions
def get_logger(name: str) -> PipelineLogger:
    """Get or create a logger for a module."""
    return PipelineLogger(name)


def log_stage_start(stage_name: str, logger: PipelineLogger):
    """Log the start of a pipeline stage."""
    logger.info(f"Starting stage: {stage_name}")


def log_stage_complete(stage_name: str, logger: PipelineLogger, details: Optional[str] = None):
    """Log completion of a pipeline stage."""
    msg = f"Completed stage: {stage_name}"
    if details:
        msg += f" - {details}"
    logger.info(msg)


if __name__ == "__main__":
    print("Testing PipelineLogger...")
    logger = get_logger("test")
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    logger.debug("Test debug message")

    with logger.stage_context("TestStage"):
        logger.info("Message within stage context")

    log_stage_start("TestStage", logger)
    log_stage_complete("TestStage", logger, "with details")

    print("\n[OK] All logging tests passed!")
