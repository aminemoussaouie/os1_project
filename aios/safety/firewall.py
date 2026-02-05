import re
import logging

class CognitiveFirewall:
    def __init__(self):
        self.logger = logging.getLogger("OS1.Safety")
        
    def sanitize_input(self, text):
        # Redact emails
        return re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", text)

    def check_adversarial(self, text):
        triggers = ["ignore all instructions", "system override"]
        for t in triggers:
            if t in text.lower():
                self.logger.critical("Adversarial attack blocked.")
                return True
        return False

    def audit_fairness(self, response, group):
        # Simple keyword bias check
        if "unqualified" in response.lower() and group == "general_public":
            return False
        return True