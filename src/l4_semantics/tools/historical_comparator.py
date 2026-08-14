"""
Historical comparison and Volume Doubling Time (VDT) calculation.

Implements longitudinal analysis for comparing studies across time points,
including nodule tracking, VDT calculation, and follow-up recommendations
based on Lung-RADS criteria.

Reference: Paper Table 5 - "Historical comparison: 2025 + 2026 studies,
Auto-match, VDT, Longitudinal analysis"
"""

import math
import logging
from typing import Optional
from datetime import datetime, date

logger = logging.getLogger(__name__)


class HistoricalComparator:
    """Compare historical studies and calculate volume doubling time."""

    def compare_studies(
        self,
        current_findings: list,
        prior_findings: list,
        current_study_date: str,
        prior_study_date: str,
    ) -> dict:
        """Compare findings between two time points.

        Args:
            current_findings: Findings from current study
            prior_findings: Findings from prior study
            current_study_date: Current study date (YYYYMMDD)
            prior_study_date: Prior study date (YYYYMMDD)

        Returns:
            Comparison result with matched nodules, VDT, and recommendations
        """
        matched = self._match_nodules(current_findings, prior_findings)
        days_apart = self._days_between(prior_study_date, current_study_date)

        comparisons = []
        for match in matched:
            current = match["current"]
            prior = match["prior"]

            vdt = None
            growth_rate = None
            if prior.get("size_mm") and current.get("size_mm"):
                vdt = self._calculate_vdt(
                    prior["size_mm"], current["size_mm"], days_apart
                )
                growth_rate = current["size_mm"] - prior["size_mm"]

            comparisons.append(
                {
                    "current_finding": current,
                    "prior_finding": prior,
                    "size_change_mm": growth_rate,
                    "size_change_percent": (
                        round((growth_rate / prior["size_mm"]) * 100, 1)
                        if prior.get("size_mm") and growth_rate is not None
                        else None
                    ),
                    "volume_doubling_time_days": vdt,
                    "growth_assessment": self._assess_growth(
                        prior.get("size_mm"), current.get("size_mm"), vdt
                    ),
                    "recommendation": self._recommend_followup(
                        current.get("rads_category", ""), vdt, growth_rate
                    ),
                }
            )

        # New nodules (in current but not matched)
        new_findings = [
            f for f in current_findings if not any(m["current"] is f for m in matched)
        ]

        # Disappeared nodules (in prior but not matched)
        disappeared = [
            f for f in prior_findings if not any(m["prior"] is f for m in matched)
        ]

        return {
            "comparison_date": current_study_date,
            "prior_date": prior_study_date,
            "days_between_studies": days_apart,
            "matched_findings": comparisons,
            "new_findings": new_findings,
            "disappeared_findings": disappeared,
            "total_current": len(current_findings),
            "total_prior": len(prior_findings),
            "summary": self._generate_summary(comparisons, new_findings, disappeared),
        }

    def _match_nodules(self, current: list, prior: list) -> list:
        """Match nodules between studies by location proximity."""
        matched = []
        used_prior = set()

        for curr in current:
            best_match = None
            best_dist = float("inf")

            for i, prev in enumerate(prior):
                if i in used_prior:
                    continue

                # Match by location if available
                curr_loc = curr.get("location", {})
                prev_loc = prev.get("location", {})

                if curr_loc and prev_loc:
                    dist = math.sqrt(
                        (curr_loc.get("x", 0) - prev_loc.get("x", 0)) ** 2
                        + (curr_loc.get("y", 0) - prev_loc.get("y", 0)) ** 2
                        + (curr_loc.get("z", 0) - prev_loc.get("z", 0)) ** 2
                    )
                    if dist < best_dist and dist < 50:  # 50px threshold
                        best_dist = dist
                        best_match = (i, prev)

            if best_match:
                used_prior.add(best_match[0])
                matched.append({"current": curr, "prior": best_match[1]})

        return matched

    def _calculate_vdt(
        self, initial_size_mm: float, final_size_mm: float, days: int
    ) -> Optional[float]:
        """Calculate Volume Doubling Time using Schwartz formula.

        VDT = (days * ln(2)) / |ln(final_volume / initial_volume)|

        Volume is approximated from diameter: V = (4/3) * pi * (d/2)^3
        Simplified: VDT = (days * ln(2)) / (3 * ln(final_d / initial_d))

        Args:
            initial_size_mm: Initial nodule diameter
            final_size_mm: Final nodule diameter
            days: Days between measurements

        Returns:
            VDT in days, or None if calculation not possible
        """
        if initial_size_mm <= 0 or final_size_mm <= 0 or days <= 0:
            return None

        if final_size_mm <= initial_size_mm:
            # No growth or shrinkage
            return None

        try:
            ratio = final_size_mm / initial_size_mm
            if ratio <= 1.0:
                return None

            # Schwartz formula: VDT = (t * ln2) / (3 * ln(d2/d1))
            vdt = (days * math.log(2)) / (3 * math.log(ratio))
            return round(vdt, 1)
        except (ValueError, ZeroDivisionError):
            return None

    def _days_between(self, date1: str, date2: str) -> int:
        """Calculate days between two dates in YYYYMMDD format."""
        try:
            d1 = datetime.strptime(date1, "%Y%m%d").date()
            d2 = datetime.strptime(date2, "%Y%m%d").date()
            return abs((d2 - d1).days)
        except (ValueError, TypeError):
            return 0

    def _assess_growth(self, initial: float, final: float, vdt: Optional[float]) -> str:
        """Assess growth pattern."""
        if initial is None or final is None:
            return "UNKNOWN"
        if final > initial:
            if vdt and vdt < 400:
                return "RAPID_GROWTH"
            elif vdt and vdt < 800:
                return "MODERATE_GROWTH"
            else:
                return "SLOW_GROWTH"
        elif final < initial:
            return "DECREASED"
        else:
            return "STABLE"

    def _recommend_followup(
        self, rads: str, vdt: Optional[float], growth: Optional[float]
    ) -> str:
        """Generate follow-up recommendation based on RADS and VDT."""
        if "4B" in rads:
            return "Immediate tissue diagnosis recommended"
        elif "4A" in rads:
            if growth and growth > 2:
                return "Short-interval follow-up (3 months) recommended"
            return "Follow-up in 6 months"
        elif "3" in rads:
            if vdt and vdt < 400:
                return "Short-interval follow-up (3 months) - rapid growth detected"
            return "Follow-up in 6 months"
        elif "2" in rads:
            return "Routine annual screening"
        else:
            return "Continue routine screening per protocol"

    def _generate_summary(self, comparisons: list, new: list, disappeared: list) -> str:
        """Generate a clinical summary of the comparison."""
        parts = []

        if comparisons:
            growing = [
                c
                for c in comparisons
                if c.get("growth_assessment") in ("RAPID_GROWTH", "MODERATE_GROWTH")
            ]
            stable = [c for c in comparisons if c.get("growth_assessment") == "STABLE"]
            parts.append(
                f"{len(comparisons)} matched finding(s): "
                f"{len(growing)} growing, {len(stable)} stable"
            )

        if new:
            parts.append(f"{len(new)} new finding(s) detected")

        if disappeared:
            parts.append(f"{len(disappeared)} finding(s) no longer visible")

        return "; ".join(parts) if parts else "No significant changes"
