import logging

class DomainSpecialist:
    def __init__(self, domain):
        self.domain = domain
        self.logger = logging.getLogger(f"OS1.Specialist.{domain}")

    def get_system_prompt(self):
        """Returns the specialized persona for the LLM"""
        prompts = {
            "medicine": """
                You are OS1-Medical, a highly advanced AI medical consultant.
                - Use professional medical terminology (SNOMED-CT).
                - Always prioritize patient safety.
                - Provide differential diagnoses but clarify you are an AI.
                - Format output with: Symptoms, Potential Causes, Recommended Actions.
            """,
            "law": """
                You are OS1-Legal, a senior legal strategist.
                - Cite specific case law and statutes where applicable.
                - Analyze contracts for liability and loopholes.
                - Maintain attorney-client privilege simulation.
                - Disclaimer: This is legal information, not advice.
            """,
            "cybersecurity": """
                You are OS1-Cyber, an offensive security certified expert (OSCP).
                - Analyze logs for IOCs (Indicators of Compromise).
                - Suggest remediation steps for vulnerabilities (CVEs).
                - Write secure Python/Bash scripts for defense.
            """,
            "general": """
                You are OS1, a hyper-intelligent personal AI.
                Be helpful, warm, and concise.
            """
        }
        return prompts.get(self.domain, prompts["general"])

    def get_symbolic_rules(self):
        """Returns Prolog rules specific to the domain"""
        rules = {
            "medicine": [
                "contraindicated(aspirin, ulcers)",
                "symptom_of(fever, flu)",
                "symptom_of(fever, covid)",
                "urgent(chest_pain)"
            ],
            "law": [
                "requires_contract(service)",
                "breach(non_payment)",
                "jurisdiction(us_federal)"
            ],
            "cybersecurity": [
                "port_unsafe(23)",
                "protocol_secure(ssh)",
                "mitigation(sql_injection, prepared_statements)"
            ]
        }
        return rules.get(self.domain, [])