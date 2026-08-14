"""Inspection router mapping DICOM metadata to clinical inspection types"""


class InspectionRouter:
    INSPECTION_MAP = [
        {
            "modality": "CT",
            "body_part": "CHEST",
            "indication": "screening",
            "type": "LUNG_CANCER_SCREENING",
        },
        {
            "modality": "CT",
            "body_part": "CHEST",
            "indication": "followup",
            "type": "NODULE_FOLLOWUP",
        },
        {
            "modality": "MG",
            "body_part": "BREAST",
            "indication": "screening",
            "type": "BREAST_SCREENING",
        },
        {
            "modality": "XR",
            "body_part": "HAND",
            "indication": "bone_age",
            "type": "BONE_AGE_ASSESSMENT",
        },
    ]

    def route(self, modality: str, body_part: str, indication: str) -> str:
        """Map DICOM metadata to clinical inspection type"""
        for rule in self.INSPECTION_MAP:
            if (
                rule["modality"] == modality
                and rule["body_part"] == body_part
                and indication in rule["indication"]
            ):
                return rule["type"]
        return "GENERAL"

    def get_supported_types(self) -> list:
        """Return all supported inspection types"""
        return [rule["type"] for rule in self.INSPECTION_MAP]

    def get_rule(self, inspection_type: str) -> dict:
        """Get routing rule for a given inspection type"""
        for rule in self.INSPECTION_MAP:
            if rule["type"] == inspection_type:
                return rule
        return None
