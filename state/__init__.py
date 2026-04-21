"""
State Package - State management for pipeline persistence
"""

from .manager import StateManager, get_state_manager

__all__ = [
    'StateManager',
    'get_state_manager'
]