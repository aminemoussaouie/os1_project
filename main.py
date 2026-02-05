from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import shutil
import os
import uvicorn
import logging

# AIOS Modules
from aios.brain.core import OS1Brain
from aios.brain.learning import RLAgent
from aios.brain.reasoning import BayesianDecision
from aios.perception.senses import Senses
from aios.perception.voice import VoiceEngine
from aios.memory.manager import MemoryManager
from aios.tools.toolbox import Toolbox
from aios.safety.firewall import CognitiveFirewall  # NEW

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OS1.Kernel")

app = FastAPI(title="OS1 Kernel")

# Initialize Subsystems
brain = OS1Brain()
senses = Senses()
memory = MemoryManager()
voice = VoiceEngine()
rl_agent = RLAgent()
bayes = BayesianDecision()
tools = Toolbox()
firewall = CognitiveFirewall() # NEW

class InteractionRequest(BaseModel):
    text: str
    user_id: str

@app.get("/")
def health_check():
    return {"status": "OS1 Online", "system": "Nominal"}

@app.post("/interact/text")
async def text_interaction(req: InteractionRequest, background_tasks: BackgroundTasks):
    logger.info(f"Processing text from {req.user_id}")

    # 0. SAFETY CHECK (Firewall)
    if firewall.check_adversarial(req.text):
        return {"response": "I cannot comply with that request due to safety protocols.", "status": "blocked"}
    
    clean_text = firewall.sanitize_input(req.text)

    # 1. MODE SWITCHING (Simple Intent Detection)
    if "medical" in clean_text.lower() or "doctor" in clean_text.lower():
        brain.switch_mode("medicine")
    elif "legal" in clean_text.lower() or "lawyer" in clean_text.lower():
        brain.switch_mode("law")
    elif "hack" in clean_text.lower() or "cyber" in clean_text.lower():
        brain.switch_mode("cybersecurity")
    else:
        brain.switch_mode("general")

    # 2. Memory Retrieval
    context = memory.retrieve_context(clean_text)
    
    # 3. Tool Check
    tool_result = ""
    if "time" in clean_text.lower():
        tool_result = tools.execute("get_time", None)
        context += f"\n[System Info: Current Time is {tool_result}]"

    # 4. Bayesian Confidence Check
    confidence = bayes.assess_confidence(len(clean_text), 0.3)
    
    # 5. Generate Response
    response_text = brain.generate_response(clean_text, context, "Neutral")

    # 6. Audit Fairness (Post-Gen Safety)
    if not firewall.audit_fairness(response_text, "general_public"):
        response_text += "\n[Audit Note: This response has been flagged for potential bias review.]"

    # 7. Generate Audio
    output_audio = "response_text.wav"
    voice.speak(response_text, output_audio)

    # 8. Background Learning & Memory
    background_tasks.add_task(rl_agent.update_policy, 0.5)
    background_tasks.add_task(memory.add_episodic_memory, clean_text, response_text, "Neutral")

    next_opt = rl_agent.get_optimization_action()

    return {
        "response": response_text,
        "audio_path": output_audio,
        "meta": {
            "confidence": confidence,
            "tool_output": tool_result,
            "optimization": next_opt
        }
    }

@app.post("/interact/audio")
async def audio_interaction(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. Save Audio
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Perception (STT)
    user_text = senses.listen_to_audio_file(temp_filename)
    logger.info(f"Heard: {user_text}")
    
    # 3. Safety Check on Transcription
    if firewall.check_adversarial(user_text):
        return {"response": "Safety protocol engaged."}
    clean_text = firewall.sanitize_input(user_text)

    # 4. Cognitive Pipeline
    context = memory.retrieve_context(clean_text)
    response_text = brain.generate_response(clean_text, context, "Audio_Input")
    
    # 5. Voice Generation (TTS)
    output_audio_path = f"response_{file.filename}.wav"
    voice.speak(response_text, output_audio_path)
    
    # 6. Cleanup & Memory
    background_tasks.add_task(os.remove, temp_filename)
    background_tasks.add_task(memory.add_episodic_memory, clean_text, response_text, "Audio")

    return {
        "transcription": clean_text,
        "response_text": response_text,
        "audio_path": output_audio_path
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)