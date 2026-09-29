"""Domain exceptions for the educational application."""


class PracticeError(Exception):
    """A recoverable user-facing error."""


class InvalidLayerSizeError(PracticeError):
    """Layer dimensions or activation are invalid."""


class MismatchedDataError(PracticeError):
    """Data do not meet the numerical contract."""


class DataFileError(PracticeError):
    """Data cannot be read or written."""
