"""tests/test_utils.py - Utility tests"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.errors import PipelineError, APIError, ValidationError


def test_exception_hierarchy():
    """Test exception inheritance and attributes."""
    err = PipelineError("Test message", stage="TestStage")
    assert err.message == "Test message"
    assert err.stage == "TestStage"
    assert err.recoverable == True
    assert str(err) == "[TestStage] Test message"


def test_api_error():
    """Test API error with context."""
    err = APIError("API failed", api_name="Cerebras", status_code=500)
    assert err.context["api_name"] == "Cerebras"
    assert err.context["status_code"] == 500
    assert err.stage == "Cerebras API"


def test_validation_error():
    """Test validation error with field and value."""
    err = ValidationError("Invalid value", field="test_field", value=42)
    assert err.context["field"] == "test_field"
    assert err.context["value"] == "42"
    assert err.recoverable == False


if __name__ == "__main__":
    print("Running utility tests...")
    test_exception_hierarchy()
    print("[OK] Exception hierarchy test passed")
    test_api_error()
    print("[OK] API error test passed")
    test_validation_error()
    print("[OK] Validation error test passed")
    print("\nAll utility tests passed!")
