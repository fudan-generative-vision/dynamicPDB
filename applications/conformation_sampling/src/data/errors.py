

"""Error class for handled errors."""


class DataError(Exception):
    """Data exception."""
    pass


class FileExistsError(DataError):
    """Raised when file already exists."""
    pass


class MmcifParsingError(DataError):
    """Raised when mmcif parsing fails."""
    pass


class ResolutionError(DataError):
    """Raised when resolution isn't acceptable."""
    pass


class LengthError(DataError):
    """Raised when length isn't acceptable."""
    pass