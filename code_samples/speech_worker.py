import speech_recognition as sr
import re
from PyQt6.QtCore import QThread, pyqtSignal

class SpeechWorker(QThread):
    """
    Hilo asíncrono para la transcripción de voz y análisis de dicción.
    Utiliza QThread para aislar el procesamiento de audio y no congelar la UI.
    """
    transcription_ready = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    diction_analyzed = pyqtSignal(int) 

    def __init__(self, language: str = "es-ES"):
        super().__init__()
        self.language = language
        self.recognizer = sr.Recognizer()
        
        # Ajuste de fluidez para pausas naturales
        self.recognizer.pause_threshold = 1.0
        self.is_listening = True
        
        # Lista de muletillas (Regex) para evaluación del candidato
        self.filler_words = [r'\beh\b', r'\bem\b', r'\bmmm\b', r'\beste\b', r'\bo sea\b', r'\bbueno\b', r'\bdigamos\b', r'\btipo\b']

    def analyze_diction(self, text: str) -> int:
        """Analiza la cantidad de muletillas en el texto transcrito (Regex)."""
        count = 0
        text_lower = text.lower()
        for pattern in self.filler_words:
            count += len(re.findall(pattern, text_lower))
        return count

    def run(self):
        with sr.Microphone() as source:
            self.status_changed.emit("🎙️ Calibrando micrófono...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_listening:
                try:
                    self.status_changed.emit("🔴 Escuchando...")
                    
                    # El timeout evita que el hilo se bloquee si hay silencio absoluto
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=None)
                    
                    if not self.is_listening:
                        break

                    self.status_changed.emit("⏳ Transcribiendo...")
                    
                    # MVP: Transcripción vía Google STT. 
                    # Roadmap: Migrar a Whisper (Local) para cumplir 100% el Privacy by Design.
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    
                    if text and self.is_listening:
                        filler_count = self.analyze_diction(text)
                        self.diction_analyzed.emit(filler_count)
                        self.transcription_ready.emit(text)
                        
                except sr.WaitTimeoutError:
                    continue # Silencio esperado (timeout), seguimos iterando
                except sr.UnknownValueError:
                    continue # Ruido ininteligible, se ignora
                except sr.RequestError as e:
                    self.error_occurred.emit(f"Error de conexión STT: {e}")
                    break
                except Exception as e:
                    self.error_occurred.emit(f"Error inesperado de micrófono: {str(e)}")
                    break
                    
        self.status_changed.emit("Inactivo")

    def stop(self):
        """Apaga el interruptor de forma segura y cierra el hilo"""
        self.is_listening = False
        self.quit()
        self.wait()