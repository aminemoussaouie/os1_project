#!/usr/bin/env python3
"""
SCRIPT-GUIDED SAMANTHA EXTRACTOR (ULTIMATE EDITION)
===================================================
1. Parses 'her_screenplay.txt' to find Samantha's exact lines.
2. Transcribes 'Her_2013.mp4' using WhisperX (word-level precision).
3. Aligns audio to script using Fuzzy Logic (Matches "Ground Truth").
4. Extracts clips.
5. Runs Demucs AI to remove Arcade Fire's score from the voice.
"""

import re
import os
import shutil
import subprocess
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

class UltimateScriptAligner:
    def __init__(self, movie_path, script_path):
        self.movie_path = Path(movie_path)
        self.script_path = Path(script_path)
        self.work_dir = Path("samantha_script_dataset")
        
        # Directory structure
        self.raw_dir = self.work_dir / "raw"
        self.clips_dir = self.work_dir / "clips_raw"
        self.clean_dir = self.work_dir / "clips_clean"
        self.final_dir = self.work_dir / "final"
        
        for d in [self.work_dir, self.raw_dir, self.clips_dir, self.clean_dir, self.final_dir]:
            d.mkdir(exist_ok=True, parents=True)
            
        self.raw_audio = self.raw_dir / "full_movie.wav"
        self.samantha_lines = []
        self.dataset_entries = []

    def check_dependencies(self):
        """Ensure the heavy artillery is installed"""
        console.print("[cyan]Checking specialized dependencies...[/cyan]")
        required = ["whisperx", "rapidfuzz", "demucs", "pydub", "torch"]
        
        missing = []
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            console.print(f"[yellow]Installing missing packages: {', '.join(missing)}...[/yellow]")
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
            
        # FFmpeg check
        if shutil.which("ffmpeg") is None:
            console.print("[red]CRITICAL: FFmpeg not found. Please install it (sudo apt install ffmpeg).[/red]")
            return False
        return True

    def parse_screenplay(self):
        """
        Parses the screenplay text to extract Samantha's Ground Truth dialogue.
        Handles IMSDb formatting.
        """
        console.print(f"\n[bold cyan]Step 1: Parsing Screenplay {self.script_path}[/bold cyan]")
        
        try:
            with open(self.script_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except FileNotFoundError:
            console.print(f"[red]Error: {self.script_path} not found. Please create it![/red]")
            return False

        is_samantha = False
        buffer_text = []
        
        # Regex for IMSDb Character Names (Centered, All Caps)
        # Matches: "                      SAMANTHA"
        name_pattern = re.compile(r"^\s{10,}SAMANTHA(?:\s\(.*\))?$")
        
        # Regex for Dialogue (Indented but less than name)
        # Matches: "            Hello, I'm here."
        dialogue_pattern = re.compile(r"^\s{5,}(.+)$")
        
        # Regex for Scene Headers or Transitions (Stops dialogue)
        stop_pattern = re.compile(r"^\s*(?:INT\.|EXT\.|CUT TO:|FADE|\[)")
        
        # Regex for Other Characters
        other_char_pattern = re.compile(r"^\s{10,}[A-Z][A-Z\s\.]+$")

        for line in lines:
            line_stripped = line.rstrip()
            if not line_stripped: continue

            # Check for Samantha header
            if name_pattern.match(line):
                if is_samantha and buffer_text:
                    self.samantha_lines.append(" ".join(buffer_text))
                    buffer_text = []
                is_samantha = True
                continue
            
            # Check for Interruption (Other character or Scene change)
            if stop_pattern.match(line) or (other_char_pattern.match(line) and "SAMANTHA" not in line):
                if is_samantha and buffer_text:
                    self.samantha_lines.append(" ".join(buffer_text))
                    buffer_text = []
                is_samantha = False
                continue

            # Collect Dialogue
            if is_samantha:
                # Remove parentheticals like (beat) or (laughing)
                clean_line = re.sub(r"\(.*?\)", "", line).strip()
                
                # Verify it's text (not empty after cleaning) and not a scene direction in caps
                if clean_line and not clean_line.isupper():
                    buffer_text.append(clean_line)

        # Catch tail
        if is_samantha and buffer_text:
            self.samantha_lines.append(" ".join(buffer_text))

        console.print(f"[green]✓ Extracted {len(self.samantha_lines)} verified Samantha lines[/green]")
        
        # Debug dump
        with open(self.work_dir / "parsed_lines_debug.txt", "w") as f:
            f.write("\n".join(self.samantha_lines))
        
        return True

    def extract_full_audio(self):
        """Extract high-fidelity audio from movie container"""
        console.print("\n[bold cyan]Step 2: Extracting Audio Track[/bold cyan]")
        
        if self.raw_audio.exists():
            console.print("[yellow]Using existing audio file[/yellow]")
            return

        # FFmpeg works on MP4, MKV, AVI, etc.
        cmd = [
            'ffmpeg', '-i', str(self.movie_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', # 16kHz for Whisper/Demucs compatibility
            '-ac', '1',     # Downmix to mono for transcription (Demucs handles stereo, but mono is safer for alignment)
            '-y', str(self.raw_audio)
        ]
        
        console.print(f"Extracting from {self.movie_path}...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print("[green]✓ Audio extracted[/green]")

    def clean_clip_with_demucs(self, input_path, clip_name):
        """
        Runs Demucs on a single clip to remove background music.
        """
        # We use demucs via subprocess to isolate the vocals
        cmd = [
            "demucs",
            "-n", "htdemucs", # High quality model
            "--two-stems", "vocals",
            str(input_path),
            "-o", str(self.clean_dir)
        ]
        
        # Suppress output
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Demucs output structure: output_dir/htdemucs/clip_name/vocals.wav
        demucs_out = self.clean_dir / "htdemucs" / clip_name / "vocals.wav"
        
        if demucs_out.exists():
            final_path = self.final_dir / f"{clip_name}.wav"
            shutil.move(str(demucs_out), str(final_path))
            # Cleanup intermediate
            shutil.rmtree(self.clean_dir / "htdemucs" / clip_name, ignore_errors=True)
            return final_path
        return None

    def align_and_extract(self):
        """
        The Core Engine:
        1. WhisperX Transcription
        2. Fuzzy Match against Script
        3. Cut Clip
        4. Clean Clip (Demucs)
        """
        import whisperx
        import torch
        from rapidfuzz import fuzz
        from pydub import AudioSegment

        console.print("\n[bold cyan]Step 3: Transcription & Alignment[/bold cyan]")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        # 1. Transcribe
        console.print(f"Loading WhisperX on {device}...")
        model = whisperx.load_model("medium.en", device, compute_type=compute_type)
        audio = whisperx.load_audio(str(self.raw_audio))
        
        console.print("Transcribing (this provides precise timestamps)...")
        result = model.transcribe(audio, batch_size=8)
        
        # 2. Align (Character level precision)
        console.print("Aligning phonemes...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device=device, return_char_alignments=False)
        
        # 3. Match and Extract
        console.print("\n[bold cyan]Step 4: Matching & Cleaning[/bold cyan]")
        full_audio_seg = AudioSegment.from_wav(self.raw_audio)
        
        matched_count = 0
        
        with Progress() as progress:
            task = progress.add_task("Processing Segments...", total=len(result["segments"]))
            
            for seg in result["segments"]:
                text = seg["text"].strip()
                if len(text) < 5: 
                    progress.update(task, advance=1)
                    continue
                
                # Fuzzy Match against Script
                best_ratio = 0
                for line in self.samantha_lines:
                    # Token Set Ratio handles partial matches well
                    ratio = fuzz.token_set_ratio(text.lower(), line.lower())
                    if ratio > best_ratio:
                        best_ratio = ratio
                
                # Strict threshold: 85% match means it's definitely her line from the script
                if best_ratio >= 85:
                    duration = seg["end"] - seg["start"]
                    # GPT-SoVITS optimal range: 2s to 15s
                    if 1.0 <= duration <= 15.0:
                        
                        # Export Raw Clip
                        clip_name = f"samantha_{matched_count:04d}"
                        raw_clip_path = self.clips_dir / f"{clip_name}.wav"
                        
                        start_ms = int(seg["start"] * 1000)
                        end_ms = int(seg["end"] * 1000)
                        
                        clip_audio = full_audio_seg[start_ms:end_ms]
                        clip_audio.export(raw_clip_path, format="wav")
                        
                        # CLEAN WITH DEMUCS (The Secret Sauce)
                        # We only clean matched clips to save massive amounts of time
                        final_path = self.clean_clip_with_demucs(raw_clip_path, clip_name)
                        
                        if final_path:
                            # Format: absolute_path|speaker|language|text
                            # We use the TRANSCRIBED text because it matches the audio exactly
                            entry = f"{final_path.absolute()}|samantha|en|{text}"
                            self.dataset_entries.append(entry)
                            matched_count += 1
                
                progress.update(task, advance=1)
            
        # Save dataset list
        list_path = self.work_dir / "filelist.list"
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.dataset_entries))
            
        console.print(f"\n[bold green]✓ Complete! Generated {matched_count} verified, cleaned clips.[/bold green]")
        console.print(f"Dataset location: {self.final_dir}")
        console.print(f"Filelist location: {list_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Script-Guided Samantha Extraction")
    parser.add_argument("movie", help="Path to Her video file (mp4/mkv)")
    parser.add_argument("script", help="Path to screenplay text file")
    
    args = parser.parse_args()
    
    if not Path(args.movie).exists():
        console.print("[red]Movie file not found.[/red]")
        return
    if not Path(args.script).exists():
        console.print("[red]Script file not found.[/red]")
        return

    extractor = UltimateScriptAligner(args.movie, args.script)
    
    if extractor.check_dependencies():
        if extractor.parse_screenplay():
            extractor.extract_full_audio()
            extractor.align_and_extract()

if __name__ == "__main__":
    main()