import sys
import os

# --- Ensure ffmpeg path is set before importing pydub ---
ffmpeg_path = r"F:\SEM-6\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\ffmpeg-2025-03-31-git-35c091f4b7-essentials_build\bin"
if os.path.exists(ffmpeg_path):
    if ffmpeg_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + ffmpeg_path
        print(f"Added {ffmpeg_path} to PATH. New PATH: {os.environ['PATH']}")
    else:
        print(f"ffmpeg path {ffmpeg_path} already in PATH")

import re
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QLineEdit, QLabel, QComboBox, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt
import whisper
from pydub import AudioSegment

class KeywordDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSLAKWS - Few-Shot Language-Agnostic Keyword Spotting System")
        self.setGeometry(100, 100, 600, 400)

        # Check if ffmpeg is available
        if not self._check_ffmpeg():
            QMessageBox.critical(self, "Error", "ffmpeg is not found. Please verify the path or reinstall ffmpeg.")
            sys.exit(1)

        # Load Whisper model
        print("Loading Whisper model...")
        self.model = whisper.load_model("small")
        print("Model loaded successfully.")

        # UI Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.upload_button = QPushButton("Upload Audio File")
        self.upload_button.clicked.connect(self.upload_file)
        layout.addWidget(self.upload_button)

        self.file_path_label = QLabel("No file selected")
        layout.addWidget(self.file_path_label)

        layout.addWidget(QLabel("Enter Keywords (comma separated):"))
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("e.g., अख्रोस, कस्मीर, हाँ")
        layout.addWidget(self.keyword_input)

        layout.addWidget(QLabel("Select Language:"))
        self.language_input = QComboBox()
        self.language_input.addItems(["Auto", "Hindi (hi)", "Marathi (mr)", "English (en)"])
        self.language_input.setCurrentText("Auto")
        layout.addWidget(self.language_input)

        self.detect_button = QPushButton("Detect Keywords")
        self.detect_button.clicked.connect(self.detect_keywords)
        self.detect_button.setEnabled(False)
        layout.addWidget(self.detect_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def _check_ffmpeg(self):
        try:
            result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"ffmpeg version check: {result.stdout.decode()}")
            return True
        except FileNotFoundError:
            print("ffmpeg not found in the system")
            return False

    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "F:/SEM-6", "Audio Files (*.wav *.mp3)")
        if file_name:
            self.audio_path = file_name
            self.file_path_label.setText(f"Selected file: {file_name}")
            self.detect_button.setEnabled(True)
            self.status_label.setText("File uploaded successfully")

    def detect_keywords(self):
        if not hasattr(self, 'audio_path'):
            self.status_label.setText("Please upload an audio file first")
            return

        self.detect_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.output_area.clear()
        self.status_label.setText("Processing...")

        keywords_input = self.keyword_input.text().strip()
        if not keywords_input:
            self.status_label.setText("Please enter at least one keyword")
            self.detect_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return

        keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]
        lang_code = None if self.language_input.currentText() == "Auto" else self.language_input.currentText().split(' (')[1].replace(')', '')

        self.output_area.append(f"Detecting keywords: {', '.join(keywords)}")
        self.output_area.append(f"Using language: {self.language_input.currentText()}")

        temp_path = None
        try:
            if self.audio_path.lower().endswith('.mp3'):
                self.output_area.append("Converting MP3 to WAV...")
                audio = AudioSegment.from_file(self.audio_path)
                temp_path = "temp.wav"
                audio.export(temp_path, format="wav")
                self.audio_path = temp_path

            self.output_area.append("Transcribing audio... (This may take a moment)")
            result = self.model.transcribe(self.audio_path, language=lang_code, verbose=False)
            segments = result["segments"]
            self.output_area.append("Transcription complete.\n")

            all_matches = {}
            for i, segment in enumerate(segments):
                text = segment["text"].lower()
                clean_text = re.sub(r'[^\w\s]', ' ', text)
                start_time = segment["start"]
                end_time = segment["end"]
                segment_text = segment["text"]

                for keyword in keywords:
                    if keyword in text:
                        keyword_pos = text.index(keyword)
                        segment_duration = end_time - start_time
                        keyword_start = start_time + (keyword_pos / len(text)) * segment_duration
                        keyword_end = keyword_start + (len(keyword) / len(text)) * segment_duration

                        if keyword not in all_matches:
                            all_matches[keyword] = []
                        all_matches[keyword].append({
                            "start": keyword_start,
                            "end": keyword_end,
                            "text": segment_text
                        })

                if len(segments) > 0:
                    self.progress_bar.setValue((i + 1) * 100 // len(segments))

            found_keywords = list(all_matches.keys())
            self.output_area.append("\nResults:")
            for keyword, matches in all_matches.items():
                self.output_area.append(f"\nKeyword '{keyword}':")
                for match in matches:
                    self.output_area.append(f"  Found at {match['start']:.2f}s to {match['end']:.2f}s: {match['text']}")

            total_keywords = len(keywords)
            found_count = len(found_keywords)
            missed_keywords = [kw for kw in keywords if kw not in found_keywords]
            accuracy = (found_count / total_keywords) * 100 if total_keywords > 0 else 0

            self.output_area.append("\n--- Model Evaluation ---")
            self.output_area.append(f"Total Keywords Entered: {total_keywords}")
            self.output_area.append(f"Keywords Detected (True Positives): {found_count}")
            self.output_area.append(f"Keywords Missed (False Negatives): {len(missed_keywords)}")
            self.output_area.append(f"Detection Accuracy: {accuracy:.2f}%")
            if missed_keywords:
                self.output_area.append(f"Missed Keywords: {', '.join(missed_keywords)}")
            else:
                self.output_area.append("No missed keywords!")

        except Exception as e:
            self.output_area.append(f"Error: {str(e)}")
            import traceback
            self.output_area.append(traceback.format_exc())

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                self.output_area.append("Temporary file removed")
            self.detect_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Ready")

# --- Main Entry Point ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = KeywordDetectorApp()
    window.show()
    sys.exit(app.exec_())
