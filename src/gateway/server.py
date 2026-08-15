"""
MCP-Medimg Server: MCP Server wrapping the Gateway with Tools, Resources, and Prompts.

Implements the MCP protocol primitives:
- Tools: medical_image_analyze, query_audit_trail, list_models, medical_image_compare
- Resources: audit://trails/{study_uid}, models://registry
- Prompts: lung_screening_protocol, breast_screening_protocol

Supports:
- Stdio transport (local development)
- Streamable HTTP transport (production, 2026 stateless spec)
"""

import asyncio
import json
import logging
import sys
import os
from typing import Optional

from mcp.server import MCPServer as FastMCP

from mcp_medimg.gateway import MCPMedimgGateway

logger = logging.getLogger(__name__)

# Initialize gateway with default config
_gateway: Optional[MCPMedimgGateway] = None


def get_gateway() -> MCPMedimgGateway:
    global _gateway
    if _gateway is None:
        pacs_url = os.environ.get("MCP_MEDIMG_PACS_URL", "http://localhost:8042")
        audit_db = os.environ.get("MCP_MEDIMG_AUDIT_DB", "reports/audit.db")
        _gateway = MCPMedimgGateway(
            config={
                "pacs_api_url": pacs_url,
                "audit_db": audit_db,
            }
        )
    return _gateway


# Create MCP server
mcp = FastMCP("MCP-Medimg Gateway")


# ============================================================================
# Tools
# ============================================================================


@mcp.tool()
def medical_image_analyze(
    studyInstanceUID: str,
    modality: str = "CT",
    bodyPart: str = "CHEST",
    clinicalIndication: str = "screening",
    priority: str = "ROUTINE",
    outputFormat: str = "DICOM_SR",
    verificationMode: str = "ADVISORY",
    operator: str = "system",
    dicom_data: dict = None,
) -> str:
    """Analyze a DICOM study with AI-assisted diagnosis.

    If aiModels is empty, L3 AIRouter auto-routes to optimal model.
    The workflow: L1 (retrieve) -> L2 (de-identify + audit) ->
    L4 (verification mode) -> L3 (route + infer + format) ->
    L4 (encode + verify) -> L2 (audit complete) -> L1 (store).

    Args:
        studyInstanceUID: DICOM Study Instance UID
        modality: Imaging modality (CT, MG, XR, MR, etc.)
        bodyPart: Body part examined (CHEST, BREAST, HAND, BRAIN, etc.)
        clinicalIndication: Clinical indication (screening, followup, emergency, bone_age)
        priority: Priority level (ROUTINE or STAT)
        outputFormat: Output format (DICOM_SR, DICOM_SEG, FHIR)
        verificationMode: Override verification mode (MANDATORY, ADVISORY, AUTO)
        operator: Operator identifier for audit trail
        dicom_data: Optional DICOM data dict for testing (bypasses PACS query)

    Returns:
        JSON string with analysis results, verification info, and audit trail
    """
    gw = get_gateway()
    result = gw.medical_image_analyze(
        studyInstanceUID=studyInstanceUID,
        modality=modality,
        bodyPart=bodyPart,
        clinicalIndication=clinicalIndication,
        priority=priority,
        outputFormat=outputFormat,
        verificationMode=verificationMode,
        operator=operator,
        dicom_data=dicom_data,
    )
    return json.dumps(result, indent=2, default=str, ensure_ascii=False)


@mcp.tool()
def query_audit_trail(
    study_uid: str = "",
    limit: int = 10,
) -> str:
    """Query the audit trail for a specific study or recent entries.

    Args:
        study_uid: DICOM Study Instance UID to look up (optional)
        limit: Maximum number of entries to return (default 10)

    Returns:
        JSON string with audit trail entries
    """
    gw = get_gateway()
    if study_uid:
        entry = gw.get_audit_trail(study_uid=study_uid)
        return json.dumps(entry, indent=2, default=str, ensure_ascii=False)
    else:
        # Return latest entry (could be extended for multiple)
        entry = gw.get_audit_trail()
        return json.dumps(entry, indent=2, default=str, ensure_ascii=False)


@mcp.tool()
def list_models() -> str:
    """List all registered AI models with their metadata.

    Returns:
        JSON string with model list including modality, body part, and version
    """
    gw = get_gateway()
    models = gw.list_registered_models()
    return json.dumps(models, indent=2, default=str, ensure_ascii=False)


@mcp.tool()
def medical_image_compare(
    current_study_uid: str,
    prior_study_uid: str,
    current_study_date: str,
    prior_study_date: str,
    modality: str = "CT",
    bodyPart: str = "CHEST",
) -> str:
    """Compare two DICOM studies for longitudinal analysis.

    Performs historical comparison between current and prior studies,
    including nodule matching, Volume Doubling Time (VDT) calculation
    via the Schwartz formula, and follow-up recommendations.

    Workflow:
    1. Analyze current study via medical_image_analyze
    2. Analyze prior study via medical_image_analyze
    3. Match findings between studies by location proximity
    4. Calculate VDT and growth assessment for matched nodules
    5. Generate follow-up recommendations based on Lung-RADS

    Args:
        current_study_uid: DICOM Study Instance UID of current study
        prior_study_uid: DICOM Study Instance UID of prior study
        current_study_date: Date of current study (YYYYMMDD)
        prior_study_date: Date of prior study (YYYYMMDD)
        modality: Imaging modality (CT, MG, etc.)
        bodyPart: Body part examined

    Returns:
        JSON string with comparison results including matched findings,
        VDT, growth assessment, and recommendations
    """
    gw = get_gateway()

    # Analyze current study
    current_result = gw.medical_image_analyze(
        studyInstanceUID=current_study_uid,
        modality=modality,
        bodyPart=bodyPart,
        clinicalIndication="followup",
        priority="ROUTINE",
        outputFormat="DICOM_SR",
    )
    current_findings = current_result.get("raw_findings", [])

    # Analyze prior study
    prior_result = gw.medical_image_analyze(
        studyInstanceUID=prior_study_uid,
        modality=modality,
        bodyPart=bodyPart,
        clinicalIndication="screening",
        priority="ROUTINE",
        outputFormat="DICOM_SR",
    )
    prior_findings = prior_result.get("raw_findings", [])

    # Compare studies
    comparison = gw.compare_studies(
        current_findings=current_findings,
        prior_findings=prior_findings,
        current_study_date=current_study_date,
        prior_study_date=prior_study_date,
    )

    return json.dumps(comparison, indent=2, default=str, ensure_ascii=False)


# ============================================================================
# Resources
# ============================================================================


@mcp.resource("audit://trails/{study_uid}")
def get_audit_resource(study_uid: str) -> str:
    """Retrieve the audit trail for a specific study as a resource.

    This resource provides access to the complete audit trail including
    AI inference metadata, verification decisions, and regulatory tags.
    """
    gw = get_gateway()
    entry = gw.get_audit_trail(study_uid=study_uid)
    return json.dumps(entry, indent=2, default=str, ensure_ascii=False)


@mcp.resource("models://registry")
def get_models_resource() -> str:
    """List all registered AI models as a resource.

    This resource exposes the current state of the AI model registry,
    including model versions, supported modalities, and performance metrics.
    """
    gw = get_gateway()
    models = gw.list_registered_models()
    return json.dumps(models, indent=2, default=str, ensure_ascii=False)


# ============================================================================
# Prompts
# ============================================================================


@mcp.prompt()
def lung_screening_protocol(
    studyInstanceUID: str,
    operator: str = "radiologist",
) -> str:
    """Generate a lung cancer screening analysis prompt.

    This prompt template guides the AI agent through a complete lung cancer
    screening workflow using MCP-Medimg, including LDCT acquisition,
    AI analysis with Lung-RADS classification, and mandatory radiologist
    verification.

    Args:
        studyInstanceUID: The DICOM Study Instance UID for the LDCT scan
        operator: The requesting operator identifier
    """
    return f"""You are performing a lung cancer screening analysis using MCP-Medimg.

Study Instance UID: {studyInstanceUID}
Operator: {operator}
Clinical Indication: Lung cancer screening (LDCT Chest)
Priority: ROUTINE
Expected Verification: MANDATORY (radiologist review required)

Please execute the following steps:

1. Call the `medical_image_analyze` tool with:
   - studyInstanceUID: "{studyInstanceUID}"
   - modality: "CT"
   - bodyPart: "CHEST"
   - clinicalIndication: "screening"
   - priority: "ROUTINE"
   - outputFormat: "DICOM_SR"
   - operator: "{operator}"

2. Review the results:
   - Check the Lung-RADS category of each finding
   - Note any nodules >= 6mm that require follow-up
   - Verify the verification_mode is MANDATORY

3. Query the audit trail using `query_audit_trail` with the study UID

4. Provide a clinical summary including:
   - Number of findings and their Lung-RADS categories
   - Recommended follow-up interval
   - Whether radiologist review is pending
"""


@mcp.prompt()
def breast_screening_protocol(
    studyInstanceUID: str,
    operator: str = "radiologist",
) -> str:
    """Generate a breast mammography screening analysis prompt.

    This prompt template guides the AI agent through a complete breast
    screening workflow using MCP-Medimg, including MG acquisition,
    AI analysis with BI-RADS classification, and verification.

    Args:
        studyInstanceUID: The DICOM Study Instance UID for the mammography
        operator: The requesting operator identifier
    """
    return f"""You are performing a breast mammography screening analysis using MCP-Medimg.

Study Instance UID: {studyInstanceUID}
Operator: {operator}
Clinical Indication: Breast cancer screening (MG Breast)
Priority: ROUTINE
Expected Verification: MANDATORY (radiologist review required)

Please execute the following steps:

1. Call the `medical_image_analyze` tool with:
   - studyInstanceUID: "{studyInstanceUID}"
   - modality: "MG"
   - bodyPart: "BREAST"
   - clinicalIndication: "screening"
   - priority: "ROUTINE"
   - outputFormat: "DICOM_SR"
   - operator: "{operator}"

2. Review the results:
   - Check the BI-RADS category of each finding
   - Note any BI-RADS 4+ findings that require biopsy
   - Verify the verification_mode is MANDATORY

3. Query the audit trail using `query_audit_trail` with the study UID

4. Provide a clinical summary including:
   - Number of findings and their BI-RADS categories
   - Recommended follow-up (biopsy vs. routine screening)
   - Whether radiologist review is pending
"""


# ============================================================================
# Server entry point
# ============================================================================


def run_stdio():
    """Run the MCP server with stdio transport (local development)."""
    mcp.run(transport="stdio")


def run_http(host: str = "0.0.0.0", port: int = 8000):
    """Run the MCP server with streamable HTTP transport.

    Supports the 2026 stateless specification via client-provided
    session IDs rather than server-side persistence.
    """
    mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    transport = os.environ.get("MCP_MEDIMG_TRANSPORT", "stdio")
    if transport == "http":
        run_http()
    else:
        run_stdio()
