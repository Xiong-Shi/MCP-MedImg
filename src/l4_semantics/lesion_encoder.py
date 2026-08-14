"""Lesion encoder for standardized grading systems (RADS, BI-RADS)"""


class LesionEncoder:
    def encode_rads(self, nodule_size: float, nodule_type: str) -> str:
        """Lung-RADS classification (version 1.1)"""
        if nodule_size < 6:
            return "Lung-RADS 1"
        elif nodule_size < 10:
            if nodule_type == "solid":
                return "Lung-RADS 2"
            else:
                return "Lung-RADS 3"
        elif nodule_size < 15:
            return "Lung-RADS 4A"
        else:
            return "Lung-RADS 4B"

    def encode_birads(
        self, mass_shape: str, mass_margin: str, calcification: str = "none"
    ) -> str:
        """BI-RADS classification (breast)"""
        if mass_shape == "round" and mass_margin == "circumscribed":
            return "BI-RADS 2"
        elif mass_shape == "irregular" or mass_margin == "spiculated":
            return "BI-RADS 4"
        elif calcification == "suspicious":
            return "BI-RADS 4"
        else:
            return "BI-RADS 3"

    def encode_birads_from_findings(self, findings: list) -> list:
        """Encode BI-RADS for a list of findings"""
        encoded = []
        for finding in findings:
            if finding.get("type") == "breast_lesion":
                birads = self.encode_birads(
                    finding.get("mass_shape", ""),
                    finding.get("mass_margin", ""),
                    finding.get("calcification", "none"),
                )
                encoded_finding = finding.copy()
                encoded_finding["rads_category"] = birads
                encoded.append(encoded_finding)
            else:
                encoded.append(finding)
        return encoded

    def encode_rads_from_findings(self, findings: list) -> list:
        """Encode Lung-RADS for a list of findings"""
        encoded = []
        for finding in findings:
            if finding.get("type") == "lung_nodule":
                rads = self.encode_rads(
                    finding.get("size_mm", 0), finding.get("nodule_type", "solid")
                )
                encoded_finding = finding.copy()
                encoded_finding["rads_category"] = rads
                encoded.append(encoded_finding)
            else:
                encoded.append(finding)
        return encoded
