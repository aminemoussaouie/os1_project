import cv2
import mediapipe as mp
import numpy as np
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import logging

class Senses:
    def __init__(self):
        self.logger = logging.getLogger("OS1.Senses")
        
        # Vision - MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Hearing - Faster Whisper (GPU optimized)
        # 'int8' is faster on CPU/RTX 3050 for inference
        self.stt_model = WhisperModel("small", device="auto", compute_type="int8")

    def analyze_visual_emotion(self, frame):
        """
        Process an image frame to detect emotional cues via landmarks.
        Returns: emotion_vector (dict)
        """
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        emotion_state = {"user_present": False, "attention": 0.0, "smile_prob": 0.0}

        if results.multi_face_landmarks:
            emotion_state["user_present"] = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Heuristic: Lips distance for smile
            lip_left = landmarks[61]
            lip_right = landmarks[291]
            smile_dist = np.linalg.norm([lip_left.x - lip_right.x, lip_left.y - lip_right.y])
            emotion_state["smile_prob"] = min(1.0, smile_dist * 2.0) # Normalized heuristic
            
        return emotion_state

    def listen_to_audio_file(self, file_path):
        """
        Transcribe audio file.
        """
        segments, info = self.stt_model.transcribe(file_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text