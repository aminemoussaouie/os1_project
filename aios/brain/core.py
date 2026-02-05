from llama_cpp import Llama
from pyswip import Prolog
import pymc as pm
import numpy as np
import yaml
import logging

# New Import for Specialists
from aios.brain.specialists import DomainSpecialist

class OS1Brain:
    def __init__(self):
        self.logger = logging.getLogger("OS1.Brain")
        with open('aios/config/config.yaml', 'r') as f:
            self.cfg = yaml.safe_load(f)

        # 1. Neural Engine (Llama 3.1)
        # n_gpu_layers=35 puts the whole model on your RTX 3050
        self.llm = Llama(
            model_path=self.cfg['models']['llm_path'],
            n_ctx=self.cfg['hardware']['ctx_size'],
            n_gpu_layers=self.cfg['hardware']['gpu_layers'],
            verbose=False
        )

        # 2. Symbolic Engine (Prolog)
        self.prolog = Prolog()
        self._init_logic()
        
        # 3. Specialist Mode (Default to General)
        self.current_specialist = DomainSpecialist("general")
        self.switch_mode("general")

    def _init_logic(self):
        # Basic logic rules for the OS
        self.prolog.assertz("critical_system(kernel)")
        self.prolog.assertz("critical_system(memory)")
        self.prolog.assertz("can_modify(root, X) :- critical_system(X)")

    def check_safety(self, action, user):
        """Symbolic Logic Check"""
        query = f"can_modify({user}, {action})"
        return list(self.prolog.query(query))

    def switch_mode(self, mode):
        """Switches the active Specialist Module"""
        try:
            self.current_specialist = DomainSpecialist(mode)
            # Inject symbolic rules specific to the domain
            for rule in self.current_specialist.get_symbolic_rules():
                # We wrap in try/except because re-asserting same rules might cause warnings
                try:
                    self.prolog.assertz(rule)
                except:
                    pass
            self.logger.info(f"Switched to {mode} mode.")
        except Exception as e:
            self.logger.error(f"Failed to switch mode: {e}")

    def generate_response(self, user_input, context, emotion_state):
        """
        Combines Prompt Engineering + Context + Logic + Specialist Persona
        """
        
        # Retrieve the specialized prompt from the current specialist
        specialist_prompt = self.current_specialist.get_system_prompt()
        
        system_prompt = f"""
        {specialist_prompt}
        
        Current Context: {context}
        Current Emotional State: {emotion_state}
        User Input: {user_input}
        
        Respond naturally, concisely, and warmly.
        """
        
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return output['choices'][0]['message']['content']