from llama_cpp import Llama
from pyswip import Prolog
import pymc as pm
import numpy as np
import yaml
import logging

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

    def _init_logic(self):
        # Basic logic rules for the OS
        self.prolog.assertz("critical_system(kernel)")
        self.prolog.assertz("critical_system(memory)")
        self.prolog.assertz("can_modify(root, X) :- critical_system(X)")

    def check_safety(self, action, user):
        """Symbolic Logic Check"""
        query = f"can_modify({user}, {action})"
        return list(self.prolog.query(query))

    def generate_response(self, user_input, context, emotion_state):
        """
        Combines Prompt Engineering + Context + Logic
        """
        system_prompt = f"""
        You are OS1, a hyper-intelligent, empathetic AI operating system.
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