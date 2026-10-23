"""Review web UI (FastAPI). Install with: pip install -e ".[web]" """

from bill_ocr.web.app import create_app

__all__ = ["create_app"]
