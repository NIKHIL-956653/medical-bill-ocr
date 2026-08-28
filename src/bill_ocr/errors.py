"""Central exception types for the bill OCR pipeline."""


class BillOCRError(Exception):
    """Base error for all pipeline failures. Catch this to catch everything ours."""


class UnsupportedFileError(BillOCRError):
    """File extension/type we don't handle (e.g. .docx, .zip)."""


class FileReadError(BillOCRError):
    """File exists but could not be opened/parsed (corrupt, locked, empty)."""