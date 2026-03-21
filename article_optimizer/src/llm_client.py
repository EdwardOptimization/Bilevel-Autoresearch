"""Re-export shared llm_client from root src/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.llm_client import call_llm, configure, get_provider_info, parse_json_response  # noqa: F401
