"""Load .env for integration tests so SEC_IDENTITY / CONGRESS_GOV_API_KEY etc.
are available without requiring shell exports."""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
