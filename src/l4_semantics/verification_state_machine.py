"""Verification state machine implementing the workflow defined in Section 3.4.4"""

from datetime import datetime


class VerificationStateMachine:
    STATES = [
        "PENDING",
        "UNDER_REVIEW",
        "REVIEWED",
        "MODIFIED",
        "CONFIRMED",
        "FINALIZED",
        "TIMEOUT",
    ]

    def __init__(self, verification_id: str, timeout_seconds: int = 300):
        self.verification_id = verification_id
        self.state = "PENDING"
        self.timeout_seconds = timeout_seconds
        self.start_time = datetime.now()
        self.history = [{"state": "PENDING", "timestamp": self.start_time.isoformat()}]

    def transition_to(self, new_state: str, reason: str = None) -> None:
        if new_state not in self.STATES:
            raise ValueError(f"Invalid state: {new_state}")

        valid_transitions = {
            "PENDING": ["UNDER_REVIEW"],
            "UNDER_REVIEW": ["REVIEWED", "TIMEOUT"],
            "REVIEWED": ["CONFIRMED", "MODIFIED"],
            "MODIFIED": ["CONFIRMED"],
            "CONFIRMED": ["FINALIZED"],
        }
        if new_state not in valid_transitions.get(self.state, []):
            raise ValueError(f"Invalid transition: {self.state} -> {new_state}")

        self.state = new_state
        self.history.append(
            {
                "state": new_state,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
            }
        )

        if new_state == "TIMEOUT":
            self._escalate()

    def _escalate(self):
        """Escalate to supervisor on timeout"""
        self.history.append(
            {
                "state": "ESCALATED",
                "timestamp": datetime.now().isoformat(),
                "reason": "Verification timeout - escalated to supervisor",
            }
        )

    def is_terminal(self) -> bool:
        """Check if current state is terminal"""
        return self.state in ["FINALIZED", "TIMEOUT"]

    def get_history(self) -> list:
        """Get full state transition history"""
        return self.history
