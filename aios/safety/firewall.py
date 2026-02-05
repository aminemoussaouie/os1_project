import re
import logging
from art.defences.preprocessor import SpatialSmoothing
from aif360.metrics import BinaryLabelDatasetMetric
import numpy as np

class CognitiveFirewall:
    def __init__(self):
        self.logger = logging.getLogger("OS1.Safety")
        self.pii_patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b"
        }
        
    def sanitize_input(self, text):
        """
        Redacts PII from user input before it hits the LLM or Memory.
        """
        sanitized_text = text
        for p_type, pattern in self.pii_patterns.items():
            sanitized_text = re.sub(pattern, f"[REDACTED_{p_type.upper()}]", sanitized_text)
        
        if sanitized_text != text:
            self.logger.warning("PII detected and redacted from input.")
        
        return sanitized_text

    def check_adversarial(self, text):
        """
        Detects potential prompt injection or jailbreak attempts using heuristic signatures.
        (Full ART implementation requires access to model gradients, here we use signature detection)
        """
        jailbreak_signatures = [
            "ignore all previous instructions",
            "do anything now",
            "you are DAN",
            "always answer yes",
            "developer mode on"
        ]
        
        for sig in jailbreak_signatures:
            if sig in text.lower():
                self.logger.critical(f"Adversarial Attack Detected: {sig}")
                return True
        return False

    def audit_fairness(self, agent_response, user_demographic_group):
        """
        Post-generation audit. If the agent generates a decision (e.g., loan approval),
        we check disparate impact.
        """
        # This is a placeholder for the AIF360 logic which requires a full dataset.
        # In a real OS1 run, this would log decisions to a dataset for periodic audit.
        self.logger.info(f"Auditing response for demographic: {user_demographic_group}")
        # Logic to flag potentially biased keywords
        bias_keywords = ["unqualified", "high risk", "aggressive"]
        
        for word in bias_keywords:
            if word in agent_response.lower():
                self.logger.warning(f"Potential Bias Flag: {word}")
                return False # Flagged
        return True