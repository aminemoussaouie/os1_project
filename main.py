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

class InteractionRequest(BaseModel):
    text: str
    user_id: str

@app.get("/")
def health_check():
    return {"status": "OS1 Online", "system": "Nominal"}

@app.post("/interact/text")
async def text_interaction(req: InteractionRequest, background_tasks: BackgroundTasks):
    logger.info(f"Processing text from {req.user_id}")

    # 1. Memory Retrieval
    context = memory.retrieve_context(req.text)
    
    # 2. Tool Check
    tool_result = ""
    if "time" in req.text.lower():
        tool_result = tools.execute("get_time", None)
        context += f"\n[System Info: Current Time is {tool_result}]"

    # 3. Bayesian Confidence Check
    confidence = bayes.assess_confidence(len(req.text), 0.3)
    
    # 4. Generate Response
    response_text = brain.generate_response(req.text, context, "Neutral")

    # 5. GENERATE AUDIO FOR TEXT INPUT (Corrected Access)
    output_audio = "response_text.wav"
    voice.speak(response_text, output_audio)

    # 6. Background Learning & Memory
    background_tasks.add_task(rl_agent.update_policy, 0.5)
    background_tasks.add_task(memory.add_episodic_memory, req.text, response_text, "Neutral")

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
    
    # 3. Cognitive Pipeline (Reuse logic)
    context = memory.retrieve_context(user_text)
    response_text = brain.generate_response(user_text, context, "Audio_Input")
    
    # 4. Voice Generation (TTS)
    output_audio_path = f"response_{file.filename}.wav"
    voice.speak(response_text, output_audio_path)
    
    # 5. Cleanup & Memory
    background_tasks.add_task(os.remove, temp_filename)
    background_tasks.add_task(memory.add_episodic_memory, user_text, response_text, "Audio")

    return {
        "transcription": user_text,
        "response_text": response_text,
        "audio_path": output_audio_path
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)