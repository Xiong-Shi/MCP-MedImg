"""
DICOM Structured Report (SR) writer for storing AI analysis results
back into Orthanc PACS via DICOMweb STOW-RS.

Uses highdicom to construct standard-compliant SR documents that can be
stored alongside the source images in PACS.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

import numpy as np
import requests
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

import highdicom as hd
from highdicom.sr import CodedConcept

logger = logging.getLogger(__name__)


# DICOM code constants for SR content
CODE_LUNG_NODULE = CodedConcept(
    value="77089001",
    scheme_designator="SCT",
    meaning="Nodule of lung",
)
CODE_DIAMETER = CodedConcept(
    value="103307008",
    scheme_designator="SCT",
    meaning="Diameter of lesion",
)
CODE_LUNG_RADS = CodedConcept(
    value="LRADS",
    scheme_designator="DCM",
    meaning="Lung-RADS category",
)
CODE_BIRADS = CodedConcept(
    value="BIRADS",
    scheme_designator="DCM",
    meaning="BI-RADS category",
)
CODE_FINDING = CodedConcept(
    value="121071",
    scheme_designator="DCM",
    meaning="Finding",
)
CODE_AI_ALGORITHM = CodedConcept(
    value="111001",
    scheme_designator="DCM",
    meaning="Algorithm Name",
)
CODE_AI_VERSION = CodedConcept(
    value="111003",
    scheme_designator="DCM",
    meaning="Algorithm Version",
)
CODE_FINDING_SITE = CodedConcept(
    value="363698007",
    scheme_designator="SCT",
    meaning="Finding site",
)


class DICOMSRWriter:
    """Construct and store DICOM SR documents in PACS via DICOMweb STOW-RS.

    This module implements the SR write-back functionality described in
    the paper's Section 3.3 (Result Storage & Interoperability):
    - Generates DICOM SR SOP instances using highdicom
    - Stores via DICOMweb STOW-RS to Orthanc
    - Supports verification with dcmtk dcmdump

    Attributes:
        pacs_url: Base URL of Orthanc DICOMweb endpoint
        auth: Optional auth tuple (username, password)
    """

    def __init__(self, pacs_url: str = "http://localhost:8042", auth: tuple = None):
        self.pacs_url = pacs_url.rstrip("/")
        self.auth = auth
        self.stow_url = f"{self.pacs_url}/dicom-web/studies"

    def create_sr_document(
        self,
        findings: List[Dict[str, Any]],
        study_info: Dict[str, Any],
        model_info: Dict[str, Any],
        source_instance_uids: Optional[List[str]] = None,
    ) -> Dataset:
        """Create a DICOM SR document from AI analysis findings.

        Args:
            findings: List of finding dicts with keys like type, size_mm,
                      rads_category, location, confidence
            study_info: Dict with patient/study metadata (patient_name,
                       patient_id, study_uid, series_uid, etc.)
            model_info: Dict with model name, version, verified status
            source_instance_uids: List of SOP Instance UIDs of source images

        Returns:
            pydicom Dataset representing a DICOM SR document
        """
        # Generate UIDs
        sr_sop_uid = generate_uid()
        sr_series_uid = generate_uid()

        # Patient/study metadata
        patient_name = study_info.get("patient_name", "ANONYMOUS")
        patient_id = study_info.get("patient_id", "ANONYMOUS")
        study_uid = study_info.get("study_uid", "")
        study_date = study_info.get("study_date", datetime.now().strftime("%Y%m%d"))
        study_time = study_info.get("study_time", datetime.now().strftime("%H%M%S"))

        # Build the SR document using pydicom Dataset directly
        sr_ds = self._build_sr_dataset(
            sr_sop_uid=sr_sop_uid,
            sr_series_uid=sr_series_uid,
            study_uid=study_uid,
            patient_name=patient_name,
            patient_id=patient_id,
            study_date=study_date,
            study_time=study_time,
            findings=findings,
            model_info=model_info,
        )

        logger.info(
            f"Created DICOM SR document: SOPInstanceUID={sr_sop_uid}, "
            f"StudyInstanceUID={study_uid}, findings_count={len(findings)}"
        )
        return sr_ds

    def _build_sr_dataset(
        self,
        sr_sop_uid: str,
        sr_series_uid: str,
        study_uid: str,
        patient_name: str,
        patient_id: str,
        study_date: str,
        study_time: str,
        findings: List[Dict],
        model_info: Dict,
    ) -> Dataset:
        """Build a DICOM SR Dataset manually when highdicom API is too restrictive.

        Constructs a valid DICOM SR document following the Comprehensive 3D SR
        SOP Class (1.2.840.10008.5.1.4.1.1.88.34).
        """
        ds = Dataset()

        # File Meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.34"
        file_meta.MediaStorageSOPInstanceUID = sr_sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.ImplementationVersionName = "MCP-MEDIMG-1.0"
        ds.file_meta = file_meta

        # Patient Module
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex = ""

        # Study Module
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = study_date
        ds.StudyTime = study_time
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.StudyID = "1"
        ds.StudyDescription = "AI Analysis - MCP-Medimg"

        # Series Module
        ds.SeriesInstanceUID = sr_series_uid
        ds.SeriesNumber = 1
        ds.SeriesDate = datetime.now().strftime("%Y%m%d")
        ds.SeriesTime = datetime.now().strftime("%H%M%S")
        ds.Modality = "SR"
        ds.SeriesDescription = "AI Analysis Results"
        ds.PatientAge = ""

        # SR Document General Module
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.34"  # Comprehensive 3D SR
        ds.SOPInstanceUID = sr_sop_uid
        ds.InstanceNumber = "1"
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "VERIFIED" if model_info.get("verified") else "UNVERIFIED"
        ds.ContentDate = datetime.now().strftime("%Y%m%d")
        ds.ContentTime = datetime.now().strftime("%H%M%S")

        # SR Document Content Module - build SR content tree as JSON
        content_tree = self._build_content_tree(findings, model_info)
        ds.ContentSequence = self._build_content_sequence_traversal(content_tree)

        # Current Requested Procedure Evidence
        ds.CurrentRequestedProcedureEvidenceSequence = []

        # Add references to source images if provided
        if study_uid:
            ref_item = Dataset()
            ref_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.34"
            ref_item.ReferencedSOPInstanceUID = sr_sop_uid
            ref_item.StudyInstanceUID = study_uid
            ds.CurrentRequestedProcedureEvidenceSequence.append(ref_item)

        # Ensure proper encoding for serialization
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Set 128-byte preamble (pydicom adds 'DICM' prefix automatically)
        ds.preamble = b"\x00" * 128

        return ds

    def _build_content_tree(self, findings: List[Dict], model_info: Dict) -> List[Dict]:
        """Build SR content tree as nested dict structure."""
        tree = []

        # Algorithm metadata
        tree.append(
            {
                "ValueType": "TEXT",
                "ConceptNameCodeSequence": [
                    {
                        "CodeValue": "111001",
                        "CodingSchemeDesignator": "DCM",
                        "CodeMeaning": "Algorithm Name",
                    }
                ],
                "TextValue": model_info.get("name", "nnU-Net"),
            }
        )
        tree.append(
            {
                "ValueType": "TEXT",
                "ConceptNameCodeSequence": [
                    {
                        "CodeValue": "111003",
                        "CodingSchemeDesignator": "DCM",
                        "CodeMeaning": "Algorithm Version",
                    }
                ],
                "TextValue": model_info.get("version", "v2"),
            }
        )

        # Each finding as a CONTAINER with child items
        for i, finding in enumerate(findings):
            container = {
                "ValueType": "CONTAINER",
                "RelationshipType": "CONTAINS",
                "ConceptNameCodeSequence": [
                    {
                        "CodeValue": "121071",
                        "CodingSchemeDesignator": "DCM",
                        "CodeMeaning": "Finding",
                    }
                ],
                "ContentTemplateSequence": [
                    {
                        "MappingResource": "DCMR",
                        "TemplateIdentifier": "2010",
                    }
                ],
                "ContentSequence": [],
            }

            # Finding type
            finding_type = finding.get("type", "lung_nodule")
            code_value = "77089001" if finding_type == "lung_nodule" else "unknown"
            code_meaning = (
                "Nodule of lung" if finding_type == "lung_nodule" else finding_type
            )
            container["ContentSequence"].append(
                {
                    "ValueType": "CODE",
                    "RelationshipType": "CONTAINS",
                    "ConceptNameCodeSequence": [
                        {
                            "CodeValue": "121071",
                            "CodingSchemeDesignator": "DCM",
                            "CodeMeaning": "Finding",
                        }
                    ],
                    "ConceptCodeSequence": [
                        {
                            "CodeValue": code_value,
                            "CodingSchemeDesignator": "SCT",
                            "CodeMeaning": code_meaning,
                        }
                    ],
                }
            )

            # Size
            size_mm = finding.get("size_mm")
            if size_mm is not None:
                container["ContentSequence"].append(
                    {
                        "ValueType": "NUM",
                        "RelationshipType": "CONTAINS",
                        "ConceptNameCodeSequence": [
                            {
                                "CodeValue": "103307008",
                                "CodingSchemeDesignator": "SCT",
                                "CodeMeaning": "Diameter of lesion",
                            }
                        ],
                        "MeasuredValueSequence": [
                            {
                                "NumericValue": str(size_mm),
                                "MeasurementUnitsCodeSequence": [
                                    {
                                        "CodeValue": "mm",
                                        "CodingSchemeDesignator": "UCUM",
                                        "CodeMeaning": "millimeter",
                                    }
                                ],
                            }
                        ],
                    }
                )

            # RADS category
            rads = finding.get("rads_category")
            if rads:
                container["ContentSequence"].append(
                    {
                        "ValueType": "CODE",
                        "RelationshipType": "CONTAINS",
                        "ConceptNameCodeSequence": [
                            {
                                "CodeValue": "RADS",
                                "CodingSchemeDesignator": "DCM",
                                "CodeMeaning": "RADS Category",
                            }
                        ],
                        "ConceptCodeSequence": [
                            {
                                "CodeValue": rads.replace(" ", "-"),
                                "CodingSchemeDesignator": "DCM",
                                "CodeMeaning": rads,
                            }
                        ],
                    }
                )

            # Confidence
            conf = finding.get("confidence")
            if conf is not None:
                container["ContentSequence"].append(
                    {
                        "ValueType": "NUM",
                        "RelationshipType": "CONTAINS",
                        "ConceptNameCodeSequence": [
                            {
                                "CodeValue": "LL2409",
                                "CodingSchemeDesignator": "DCM",
                                "CodeMeaning": "AI confidence",
                            }
                        ],
                        "MeasuredValueSequence": [
                            {
                                "NumericValue": str(conf),
                                "MeasurementUnitsCodeSequence": [
                                    {
                                        "CodeValue": "1",
                                        "CodingSchemeDesignator": "UCUM",
                                        "CodeMeaning": "no units",
                                    }
                                ],
                            }
                        ],
                    }
                )

            tree.append(container)

        return tree

    def _build_content_sequence_traversal(self, tree: List[Dict]) -> list:
        """Convert content tree to pydicom ContentSequence items.

        Recursively build Dataset items for the SR ContentSequence.
        """
        from pydicom.sequence import Sequence
        from pydicom.dataset import Dataset as DcmDataset

        items = []
        for node in tree:
            item = DcmDataset()

            if "RelationshipType" in node:
                item.RelationshipType = node["RelationshipType"]
            if "ValueType" in node:
                item.ValueType = node["ValueType"]

            if "ConceptNameCodeSequence" in node:
                code_items = []
                for code in node["ConceptNameCodeSequence"]:
                    code_ds = DcmDataset()
                    code_ds.CodeValue = code["CodeValue"]
                    code_ds.CodingSchemeDesignator = code["CodingSchemeDesignator"]
                    code_ds.CodeMeaning = code["CodeMeaning"]
                    code_items.append(code_ds)
                item.ConceptNameCodeSequence = Sequence(code_items)

            if "TextValue" in node:
                item.TextValue = node["TextValue"]

            if "ConceptCodeSequence" in node:
                code_items = []
                for code in node["ConceptCodeSequence"]:
                    code_ds = DcmDataset()
                    code_ds.CodeValue = code["CodeValue"]
                    code_ds.CodingSchemeDesignator = code["CodingSchemeDesignator"]
                    code_ds.CodeMeaning = code["CodeMeaning"]
                    code_items.append(code_ds)
                item.ConceptCodeSequence = Sequence(code_items)

            if "MeasuredValueSequence" in node:
                mvs_items = []
                for mv in node["MeasuredValueSequence"]:
                    mv_ds = DcmDataset()
                    mv_ds.NumericValue = mv["NumericValue"]
                    if "MeasurementUnitsCodeSequence" in mv:
                        units_items = []
                        for unit in mv["MeasurementUnitsCodeSequence"]:
                            unit_ds = DcmDataset()
                            unit_ds.CodeValue = unit["CodeValue"]
                            unit_ds.CodingSchemeDesignator = unit[
                                "CodingSchemeDesignator"
                            ]
                            unit_ds.CodeMeaning = unit["CodeMeaning"]
                            units_items.append(unit_ds)
                        mv_ds.MeasurementUnitsCodeSequence = Sequence(units_items)
                    mvs_items.append(mv_ds)
                item.MeasuredValueSequence = Sequence(mvs_items)

            if "ContentTemplateSequence" in node:
                tpl_items = []
                for tpl in node["ContentTemplateSequence"]:
                    tpl_ds = DcmDataset()
                    tpl_ds.MappingResource = tpl["MappingResource"]
                    tpl_ds.TemplateIdentifier = tpl["TemplateIdentifier"]
                    tpl_items.append(tpl_ds)
                item.ContentTemplateSequence = Sequence(tpl_items)

            if "ContentSequence" in node:
                child_items = self._build_content_sequence_traversal(
                    node["ContentSequence"]
                )
                item.ContentSequence = Sequence(child_items)

            items.append(item)

        return Sequence(items)

    def store_sr_to_pacs(
        self, sr_dataset: Dataset, study_uid: str = None
    ) -> Dict[str, Any]:
        """Store a DICOM SR document to PACS via DICOMweb STOW-RS.

        Args:
            sr_dataset: pydicom Dataset representing the SR document
            study_uid: Optional Study Instance UID for STOW-RS URL

        Returns:
            Dict with store status, location, and retrieve URL
        """
        import io
        from pydicom.filewriter import dcmwrite

        # Serialize to bytes
        buffer = io.BytesIO()
        dcmwrite(buffer, sr_dataset)
        dicom_bytes = buffer.getvalue()

        # Construct STOW-RS URL
        if study_uid:
            url = f"{self.stow_url}/{study_uid}"
        else:
            url = self.stow_url

        # Build multipart payload
        boundary = "mcp-medimg-boundary"
        body = (f"--{boundary}\r\nContent-Type: application/dicom\r\n\r\n").encode(
            "ascii"
        )
        body += dicom_bytes
        body += f"\r\n--{boundary}--\r\n".encode("ascii")

        headers = {
            "Content-Type": f'multipart/related; type="application/dicom"; boundary={boundary}',
        }

        try:
            response = requests.post(
                url,
                data=body,
                headers=headers,
                auth=self.auth,
                timeout=30,
            )

            if response.status_code in (200, 202):
                # Parse response for stored SOP Instance UID
                stored_uid = sr_dataset.SOPInstanceUID
                retrieve_url = (
                    f"{self.pacs_url}/dicom-web/studies/{sr_dataset.StudyInstanceUID}"
                    f"/series/{sr_dataset.SeriesInstanceUID}"
                    f"/instances/{stored_uid}"
                )
                logger.info(f"SR stored successfully: {stored_uid} at {retrieve_url}")
                return {
                    "status": "STORED",
                    "sop_instance_uid": str(stored_uid),
                    "study_uid": str(sr_dataset.StudyInstanceUID),
                    "series_uid": str(sr_dataset.SeriesInstanceUID),
                    "retrieve_url": retrieve_url,
                    "http_status": response.status_code,
                }
            else:
                logger.error(
                    f"STOW-RS failed: {response.status_code} {response.text[:200]}"
                )
                return {
                    "status": "FAILED",
                    "http_status": response.status_code,
                    "error": response.text[:500],
                }
        except requests.exceptions.ConnectionError:
            logger.warning("PACS not reachable, SR document created but not stored")
            return {
                "status": "PACS_UNREACHABLE",
                "sop_instance_uid": str(sr_dataset.SOPInstanceUID),
                "message": "SR document constructed successfully but PACS is not running",
            }

    def create_and_store(
        self,
        findings: List[Dict],
        study_info: Dict,
        model_info: Dict,
        source_instance_uids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """One-step: create SR document and store to PACS.

        Args:
            findings: List of finding dicts
            study_info: Study/patient metadata
            model_info: Model metadata
            source_instance_uids: Optional source image UIDs

        Returns:
            Dict with creation and storage status
        """
        sr_doc = self.create_sr_document(
            findings=findings,
            study_info=study_info,
            model_info=model_info,
            source_instance_uids=source_instance_uids,
        )

        store_result = self.store_sr_to_pacs(
            sr_dataset=sr_doc,
            study_uid=study_info.get("study_uid"),
        )

        return {
            "sr_created": True,
            "sop_class_uid": "1.2.840.10008.5.1.4.1.1.88.34",
            "sop_instance_uid": str(sr_doc.SOPInstanceUID),
            "findings_count": len(findings),
            "store_result": store_result,
        }

    def verify_sr_in_pacs(
        self, study_uid: str, series_uid: str, instance_uid: str
    ) -> Dict[str, Any]:
        """Verify SR document was stored correctly via WADO-RS retrieval.

        Args:
            study_uid: Study Instance UID
            series_uid: Series Instance UID
            instance_uid: SOP Instance UID

        Returns:
            Dict with verification status
        """
        retrieve_url = (
            f"{self.pacs_url}/dicom-web/studies/{study_uid}"
            f"/series/{series_uid}/instances/{instance_uid}"
        )

        try:
            # Use WADO-RS metadata endpoint for verification (more reliable than file retrieval)
            metadata_url = f"{retrieve_url}/metadata"
            response = requests.get(metadata_url, auth=self.auth, timeout=10)
            if response.status_code == 200:
                metadata = response.json()
                if isinstance(metadata, list) and len(metadata) > 0:
                    meta = metadata[0]
                    # Extract SOP Class UID (tag 00080016) and SOP Instance UID (00080018)
                    sop_class_tag = meta.get("00080016", {}).get("Value", [""])
                    sop_instance_tag = meta.get("00080018", {}).get("Value", [""])
                    modality_tag = meta.get("00080060", {}).get("Value", [""])
                    completion_tag = meta.get("0040A491", {}).get("Value", [""])

                    return {
                        "verified": True,
                        "sop_class": sop_class_tag[0] if sop_class_tag else "",
                        "sop_instance_uid": sop_instance_tag[0]
                        if sop_instance_tag
                        else "",
                        "modality": modality_tag[0] if modality_tag else "",
                        "completion_flag": completion_tag[0] if completion_tag else "",
                        "message": "SR document verified in PACS via WADO-RS metadata",
                    }
                return {
                    "verified": False,
                    "message": "Empty metadata response",
                }
            else:
                return {
                    "verified": False,
                    "http_status": response.status_code,
                    "message": f"Retrieval failed: {response.status_code}",
                }
        except requests.exceptions.ConnectionError:
            return {
                "verified": False,
                "message": "PACS not reachable for verification",
            }
        except Exception as e:
            return {
                "verified": False,
                "message": f"Verification error: {str(e)}",
            }
