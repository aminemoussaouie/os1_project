import os
import subprocess
import logging

class VoiceEngine:
    def __init__(self):
        self.logger = logging.getLogger("OS1.Voice")
        
        self.base_dir = os.getcwd()
        self.piper_dir = os.path.join(self.base_dir, "piper")
        self.binary_path = os.path.join(self.piper_dir, "piper")
        self.model_path = os.path.join(self.base_dir, "models", "tts", "en_US-lessac-medium.onnx")
        
        # Verify files
        if not os.path.exists(self.binary_path):
            self.logger.error(f"Piper binary missing at: {self.binary_path}")

    def speak(self, text, output_file="output.wav"):
        self.logger.info(f"Synthesizing voice...")
        
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.base_dir, output_file)

        clean_text = text.replace('"', '').replace("'", "").replace("\n", " ")
        
        # KEY FIX: Add the piper directory to LD_LIBRARY_PATH for this specific command
        # This tells Linux: "Look for .so files in the piper folder"
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.piper_dir}:{env.get('LD_LIBRARY_PATH', '')}"

        try:
            command = [
                self.binary_path,
                "--model", self.model_path,
                "--output_file", output_file
            ]
            
            process = subprocess.Popen(
                command, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd=self.piper_dir, # Run inside piper dir
                env=env             # Pass the fixed environment variables
            )
            
            stdout, stderr = process.communicate(input=clean_text.encode('utf-8'))
            
            if process.returncode == 0:
                self.logger.info(f"Audio generated: {output_file}")
            else:
                self.logger.error(f"Piper failed: {stderr.decode()}")
                
        except Exception as e:
            self.logger.error(f"TTS Execution Error: {e}")
            
        return output_file