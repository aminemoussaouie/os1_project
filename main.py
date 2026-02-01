from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from aios.brain.core import OS1Brain
from aios.perception.senses import Senses
from aios.memory.manager import MemoryManager
import shutil
import os
import uvicorn

app = FastAPI(title="OS1 Kernel")

# Initialize Systems
brain = OS1Brain()
senses = Senses()
memory = MemoryManager()

class TextRequest(BaseModel):
    text: str
    user_id: str

@app.post("/interact/text")
def text_interaction(req: TextRequest):
    # 1. Retrieve Context
    context = memory.retrieve_context(req.text)
    
    # 2. Generate Thought
    response = brain.generate_response(req.text, context, "Neutral")
    
    # 3. Consolidate Memory
    memory.add_episodic_memory(req.text, response, "Neutral")
    
    return {"response": response}

@app.post("/interact/voice")
def voice_interaction(file: UploadFile = File(...)):
    # 1. Save Audio
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Transcribe (Faster Whisper)
    text = senses.listen_to_audio_file(temp_filename)
    
    # 3. Process
    context = memory.retrieve_context(text)
    response = brain.generate_response(text, context, "Audio_Input")
    
    # 4. Clean up
    os.remove(temp_filename)
    
    return {"transcription": text, "response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)