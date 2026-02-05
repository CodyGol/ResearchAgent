"""Custom JSON serialization utilities for datetime and other complex types."""

import json
from datetime import datetime
from typing import Any


class DateTimeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj: Any) -> Any:
        """Convert datetime objects to ISO format strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def serialize_for_db(data: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize a dictionary for database storage, converting datetime objects to ISO strings.
    
    Args:
        data: Dictionary potentially containing datetime objects
        
    Returns:
        Dictionary with datetime objects converted to ISO strings
    """
    # Convert to JSON string and back to ensure all datetime objects are serialized
    json_str = json.dumps(data, cls=DateTimeJSONEncoder)
    return json.loads(json_str)
