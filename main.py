import json
import time
import os
import pyttsx3
import ollama
import speech_recognition as sr
import subprocess
import sys

class OutdoorVoiceAssistant:
    def __init__(self):
        # Check if Ollama is running
        self._check_ollama()
        
        # Initialize TTS engine first
        self.tts_engine = pyttsx3.init()
        
        # Initialize other components
        self._init_voice_models()
        self.setup_audio_devices()
        self.running = True
        # Initialize Ollama model
        self.model = "gemma3:1b"  # Using Gemma as it's lightweight and capable

        # Say hello when starting up
        self._say_hello()

    def _check_ollama(self):
        """Check if Ollama is running and available"""
        try:
            # Try to list models to check if Ollama is running
            ollama.list()
            print("✓ Ollama is running and accessible")
        except Exception as e:
            print("\n❌ Ollama is not running or not accessible!")
            print("\nPlease follow these steps:")
            print("1. Download Ollama from https://ollama.com/download")
            print("2. Install and run Ollama")
            print("3. Pull the required model by running:")
            print("   ollama pull gemma3:1b")
            print("\nAfter completing these steps, restart this program.")
            sys.exit(1)

    def _init_voice_models(self):
        """Initialize speech recognition"""
        self.recognizer = sr.Recognizer()
        # Adjust for ambient noise
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000  # Adjust this value based on your environment
        self.recognizer.pause_threshold = 0.8  # Shorter pause threshold for more responsive recognition

        # Configure TTS engine
        self.tts_engine.setProperty('rate', 150)  # Speech speed
        self.tts_engine.setProperty('volume', 1.0)  # Volume level
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[1].id)  # Select female voice

    def setup_audio_devices(self):
        """Configure audio devices"""
        # Use default microphone
        self.microphone = sr.Microphone()
        # print(f"\nUsing default microphone: {self.microphone.name}\n")

    def generate_response(self, query):
        """Generate response using local Ollama model"""
        try:
            response = ollama.generate(
                model=self.model,
                prompt=f"""You are an outdoor voice assistant. Keep your responses brief .
                Current query: {query}
                Remember to answer the question without any preamble or introduction.""",
                stream=False,
                options={
                    "temperature": 0.7,
                    "num_predict": 50  # Limit response length
                }
            )
            return response['response'].strip()
        except Exception as e:
            error_msg = str(e)
            if "Failed to connect" in error_msg:
                print("\n❌ Ollama connection error!")
                print("Please make sure Ollama is running and try again.")
            else:
                print(f"\n❌ Error generating response: {error_msg}")
            return "I'm having trouble thinking right now. Could you try again?"

    def _say_hello(self):
        """Generate and speak a welcome message"""
        try:
            welcome_response = ollama.generate(
                model=self.model,
                prompt="""Generate a very brief welcome message for an outdoor voice assistant. 
                Keep it to one short sentence. Be friendly but concise.""",
                stream=False,
                options={
                    "temperature": 0.7,
                    "num_predict": 30  # Limit response length
                }
            )
            welcome_message = welcome_response['response'].strip()
            print(f"Assistant: {welcome_message}")
            self.speak(welcome_message)
        except Exception as e:
            print(f"Error generating welcome message: {str(e)}")
            self.speak("Hello! I'm your outdoor assistant. How can I help you today?")

    def speak(self, text):
        """Convert text to speech using pyttsx3"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def start(self):
        """Start the assistant in a single loop"""
        print("Starting assistant...")
        with self.microphone as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.running:
                try:
                    print("Adjusting for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    print("\nListening...")
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                    
                    try:
                        text = self.recognizer.recognize_google(audio)
                        if text and text.strip():
                            print(f"Heard: {text}")
                            
                            # Generate and speak response
                            response = self.generate_response(text.strip())
                            print(f"Response: {response}")
                            self.speak(response)
                            
                    except sr.UnknownValueError:
                        continue  # Skip silently if no speech detected
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")
                        continue
                        
                except Exception as e:
                    if "timeout" in str(e).lower():
                        continue  # Skip silently on timeout
                    print(f"Error: {str(e)}")
                    continue

if __name__ == "__main__":
    assistant = OutdoorVoiceAssistant()
    try:
        assistant.start()
    except KeyboardInterrupt:
        print("\nStopping assistant...")