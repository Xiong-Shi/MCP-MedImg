"""PACS client for DICOMweb operations"""

import requests
from typing import Dict, List


class PACSClient:
    def __init__(self, base_url: str = "http://localhost:8042"):
        self.base_url = base_url
        self.dicomweb_url = f"{base_url}/dicom-web"

    def query_studies(self, modality: str = None) -> List[Dict]:
        params = {}
        if modality:
            params["Modality"] = modality
        response = requests.get(f"{self.dicomweb_url}/studies", params=params)
        if response.status_code == 200:
            return response.json()
        return []

    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/system", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
