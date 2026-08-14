"""PHI de-identification and security controls"""

import hashlib
import secrets
from typing import Dict


class SecurityModule:
    def __init__(self, level: str = "patient"):
        self.level = level
        self.salt = secrets.token_hex(16)

    def deidentify(self, dicom_data: Dict) -> Dict:
        result = dicom_data.copy()
        if "PatientName" in result:
            result["PatientName"] = "ANONYMIZED"
        if "PatientID" in result:
            original = result["PatientID"]
            hashed = hashlib.sha256(f"{self.salt}{original}".encode()).hexdigest()
            result["PatientID"] = f"sha256:{hashed[:16]}"
        if self.level == "patient" and "StudyDate" in result:
            result["StudyDate"] = "REDACTED"
        return result

    def hash_content(self, content: str) -> str:
        return f"sha256:{hashlib.sha256(str(content).encode()).hexdigest()}"
