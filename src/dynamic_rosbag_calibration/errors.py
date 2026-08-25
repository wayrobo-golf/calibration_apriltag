"""Stable error types and machine-readable error codes."""


class CalibrationError(RuntimeError):
    code = "CAL-E-INTERNAL"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ConfigError(CalibrationError):
    code = "CAL-E-CONFIG-SCHEMA"


class StateConflictError(CalibrationError):
    code = "CAL-E-STATE-CONFLICT"


class RecorderError(CalibrationError):
    code = "CAL-E-RECORDER"


class IntegrityError(CalibrationError):
    code = "CAL-E-CHECKSUM"
