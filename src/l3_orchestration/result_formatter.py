"""Result formatter for standardized DICOM SR/SEG output"""

import hashlib
import json
from datetime import datetime
from pydicom.uid import generate_uid


class ResultFormatter:
    def to_dicom_sr(self, findings: list, study_info: dict, model_info: dict) -> dict:
        """Format findings as DICOM SR structure

        Produces a structured report following DICOM SR SOP Class.
        In local validation, returns a dict representation that
        can be serialized to DICOM SR via highdicom.
        """
        sr_document = {
            "sopClassUID": "1.2.840.10008.5.1.4.1.1.88.22",
            "sopInstanceUID": generate_uid(),
            "seriesInstanceUID": generate_uid(),
            "studyInstanceUID": study_info.get("study_uid", ""),
            "completionFlag": "COMPLETE",
            "verificationFlag": "VERIFIED"
            if model_info.get("verified")
            else "UNVERIFIED",
            "contentDate": datetime.now().strftime("%Y%m%d"),
            "contentTime": datetime.now().strftime("%H%M%S"),
            "modelInfo": {
                "name": model_info.get("name", "unknown"),
                "version": model_info.get("version", "unknown"),
            },
            "findings": findings,
        }
        return sr_document

    def to_dicom_seg(
        self, segmentation_mask, source_images: list, segment_description: dict
    ) -> dict:
        """Format segmentation as DICOM SEG structure

        In local validation, returns a dict representation.
        Full highdicom integration is available for production use.
        """
        seg_doc = {
            "sopClassUID": "1.2.840.10008.5.1.4.1.1.66.4",
            "sopInstanceUID": generate_uid(),
            "seriesInstanceUID": generate_uid(),
            "studyInstanceUID": source_images[0].get("study_uid", "")
            if source_images
            else "",
            "segmentDescription": segment_description,
            "maskShape": list(segmentation_mask.shape)
            if hasattr(segmentation_mask, "shape")
            else None,
            "algorithmName": segment_description.get("algorithm_name", "nnU-Net"),
            "algorithmVersion": segment_description.get("algorithm_version", "v2"),
        }
        return seg_doc

    def to_fhir_observation(self, findings: list, study_info: dict) -> dict:
        """Format findings as FHIR Observation resource"""
        observations = []
        for finding in findings:
            obs = {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": finding.get("loinc_code", "18748-4"),
                            "display": finding.get(
                                "description", "Diagnostic imaging study"
                            ),
                        }
                    ]
                },
                "subject": {
                    "reference": f"Patient/{study_info.get('patient_hash', 'unknown')}"
                },
                "effectiveDateTime": datetime.now().isoformat(),
                "valueString": json.dumps(finding),
                "component": [],
            }
            if "rads_category" in finding:
                obs["component"].append(
                    {
                        "code": {
                            "coding": [
                                {
                                    "system": "http://radlex.org",
                                    "code": finding["rads_category"],
                                    "display": "RADS Category",
                                }
                            ]
                        },
                        "valueString": finding["rads_category"],
                    }
                )
            observations.append(obs)
        return observations

    def hash_result(self, result: dict) -> str:
        """Generate cryptographic hash of result for audit trail"""
        content = json.dumps(result, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
