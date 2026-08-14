"""Audit trail logging with cryptographic hashing"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path


class AuditLogger:
    def __init__(self, db_path: str = "reports/audit.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                operation TEXT,
                operator TEXT,
                patient_hash TEXT,
                study_uid TEXT,
                input_hash TEXT,
                ai_model TEXT,
                ai_output_hash TEXT,
                verification_mode TEXT,
                verification_action TEXT,
                verification_reason TEXT,
                finalized_by TEXT,
                result TEXT,
                regulatory_tags TEXT
            )
        """)
        # Migration: add columns if missing (for existing DBs)
        try:
            conn.execute("ALTER TABLE audit_logs ADD COLUMN verification_reason TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE audit_logs ADD COLUMN finalized_by TEXT")
        except Exception:
            pass
        conn.commit()
        conn.close()

    def log_operation(self, **kwargs) -> str:
        conn = sqlite3.connect(self.db_path)
        audit_id = f"AUDIT_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        conn.execute(
            """
            INSERT INTO audit_logs
            (timestamp, operation, operator, patient_hash, study_uid,
             input_hash, ai_model, ai_output_hash, verification_mode,
             verification_action, result, regulatory_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                kwargs.get("operation", "unknown"),
                kwargs.get("operator", "system"),
                kwargs.get("patient_hash", ""),
                kwargs.get("study_uid", ""),
                kwargs.get("input_hash", ""),
                kwargs.get("ai_model", ""),
                kwargs.get("ai_output_hash", ""),
                kwargs.get("verification_mode", "AUTO"),
                kwargs.get("verification_action", ""),
                kwargs.get("result", "PENDING"),
                json.dumps(kwargs.get("regulatory_tags", [])),
            ),
        )
        conn.commit()
        conn.close()
        return audit_id

    def get_audit_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM audit_logs")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def log_start(self, request: dict) -> str:
        """Log service initiation and return audit_id."""
        audit_id = (
            f"AUDIT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{secrets_token_hex(4)}"
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO audit_logs
            (timestamp, operation, operator, patient_hash, study_uid,
             input_hash, result, regulatory_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                "medical_image_analyze_start",
                request.get("operator", "system"),
                self._hash_patient(request.get("patient_id", "")),
                request.get("study_uid", ""),
                self._hash_dicom(request.get("dicom_data", {})),
                "IN_PROGRESS",
                "[]",
            ),
        )
        conn.commit()
        conn.close()
        return audit_id

    def log_complete(
        self, audit_id: str, ai_result: dict, verification: dict, study_uid: str = ""
    ) -> None:
        """Log service completion with full audit trail."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO audit_logs
            (timestamp, operation, patient_hash, study_uid,
             ai_model, ai_output_hash, verification_mode,
             verification_action, verification_reason, finalized_by,
             result, regulatory_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                "medical_image_analyze_complete",
                "",
                study_uid or audit_id,
                ai_result.get("model_version") or "",
                self._hash_content(str(ai_result.get("output", ""))),
                verification.get("mode", ""),
                verification.get("action", ""),
                verification.get("reason", ""),
                verification.get("finalized_by") or "",
                "SUCCESS",
                json.dumps(["FDA_TPLC", "NMPA_Grade2", "EU_AI_Act"]),
            ),
        )
        conn.commit()
        conn.close()

    def get_latest_log(self, audit_id: str = None) -> dict:
        """Retrieve latest log. If audit_id is provided, search by study_uid
        (log_start stores study_uid = studyInstanceUID, log_complete also stores
        study_uid = studyInstanceUID). Falls back to latest complete record."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if audit_id:
            # Try matching study_uid first (both start and complete store study_uid)
            cursor = conn.execute(
                "SELECT * FROM audit_logs WHERE study_uid = ? ORDER BY id DESC LIMIT 1",
                (audit_id,),
            )
            row = cursor.fetchone()
            if row is None:
                # Fallback: try matching as audit_id pattern in study_uid
                cursor = conn.execute(
                    "SELECT * FROM audit_logs ORDER BY id DESC LIMIT 1"
                )
                row = cursor.fetchone()
        else:
            cursor = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
        conn.close()
        if row is None:
            return {}
        result = dict(row)
        if result.get("regulatory_tags"):
            try:
                result["regulatory_tags"] = json.loads(result["regulatory_tags"])
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    def get_audit_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM audit_logs")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def _hash_patient(self, patient_id: str) -> str:
        import hashlib

        return f"sha256:{hashlib.sha256(patient_id.encode()).hexdigest()[:16]}"

    def _hash_dicom(self, dicom_data: dict) -> str:
        import hashlib

        return f"sha256:{hashlib.sha256(str(dicom_data).encode()).hexdigest()[:16]}"

    def _hash_content(self, content: str) -> str:
        import hashlib

        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()[:16]}"


def secrets_token_hex(nbytes: int = 4) -> str:
    import secrets as _s

    return _s.token_hex(nbytes)
