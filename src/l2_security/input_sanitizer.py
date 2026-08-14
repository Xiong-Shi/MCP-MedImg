"""Input sanitization to mitigate MCP-38 injection threats"""

import re


class InputSanitizer:
    def sanitize_dicom(self, dicom_data: dict) -> dict:
        """Remove or escape potentially malicious content"""
        sanitized = dicom_data.copy()
        for field in ["PatientName", "StudyDescription", "SeriesDescription"]:
            if field in sanitized:
                sanitized[field] = self._sanitize_text(sanitized[field])
        return sanitized

    def _sanitize_text(self, text: str) -> str:
        """Strip executable patterns and control characters"""
        if not isinstance(text, str):
            return text
        # Remove shell metacharacters
        text = re.sub(r"[;&$|`<>]", "", text)
        # Escape DICOM control sequences
        return text.encode("ascii", errors="ignore").decode("ascii")

    def sanitize_tool_input(self, tool_name: str, params: dict) -> dict:
        """Sanitize tool invocation parameters"""
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, str):
                sanitized[key] = self._sanitize_text(value)
            else:
                sanitized[key] = value
        return sanitized
