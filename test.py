import subprocess
import asyncio
import shlex
import re
import ollama
import sounddevice as sd
import numpy as np
from pynput import keyboard as pynput_keyboard
import threading
import queue
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Setup audio parameters
sample_rate = 16000  # Sample rate expected by Whisper
channels = 1
dtype = 'float32'


# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

# Create a queue to store audio chunks
audio_queue = queue.Queue()
is_recording = False
stop_recording = False
# Async TTS function using Piper in background
async def speak_from_queue(queue):
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        print(f"\n[SPEAKING]: {sentence}")
        safe_text = shlex.quote(sentence)
        command = (
            f"echo {safe_text} | piper --model en_US-lessac-medium.onnx --output-raw | "
            f"aplay -r 22050 -f S16_LE -t raw -"
        )
        subprocess.run(command, shell=True)
        queue.task_done()

# Main function: stream from Ollama, buffer sentences, and queue them
async def stream_and_speak(query):
    queue = asyncio.Queue()
    speaker_task = asyncio.create_task(speak_from_queue(queue))

    response = ollama.generate(
        model="gemma3:1b",
        prompt=(
            "You are an outdoor voice assistant. Be precise but short in your answers.\n"
            "Current query: {query}\n"
            "Remember to answer the question without any preamble or introduction."
        ),
        stream=True,
        options={"temperature": 0.7, "num_predict": 100}
    )

    buffer = ""
    sentence_endings = re.compile(r'([.!?])')

    for chunk in response:
        buffer += chunk["response"]
        while True:
            match = sentence_endings.search(buffer)
            if not match:
                break
            end_idx = match.end()
            complete_sentence = buffer[:end_idx].strip()
            buffer = buffer[end_idx:]
            await queue.put(complete_sentence)

    # Speak any leftover text
    if buffer.strip():
        await queue.put(buffer.strip())

    # Signal the speaker to finish
    await queue.put(None)
    await speaker_task  # Wait for speaker to finish


def audio_callback(indata, frames, time, status):
    """This is called for each audio block"""
    if is_recording:
        audio_queue.put(indata.copy())
        # print(f"Captured audio chunk of shape {indata.shape}")

def key_monitor():
    """Monitor space bar and ESC using pynput"""
    global is_recording, stop_recording

    def on_press(key):
        global is_recording
        try:
            # Check if key is the space key using pynput's Key.space
            if key == pynput_keyboard.Key.ctrl and not is_recording:
                is_recording = True
                audio_queue.queue.clear()
                print("Recording... (holding space)")
        except AttributeError:
            pass

    def on_release(key):
        global is_recording, stop_recording
        try:
            if key == pynput_keyboard.Key.ctrl and is_recording:
                is_recording = False
                print("Processing...")
                process_audio()
        except AttributeError:
            if key == pynput_keyboard.Key.esc:
                stop_recording = True
                is_recording = False
                print("Stopping...")
                return False  # Stop listener

    print("Press and hold SPACE to record. Release to transcribe. Press ESC to quit.")
    with pynput_keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def process_audio():
    """Process collected audio chunks and transcribe"""
    if audio_queue.empty():
        print("No audio recorded")
        return
    
    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())
    
    if not chunks:
        return
        
    audio_data = np.concatenate(chunks, axis=0)
    audio_flat = audio_data.flatten()
    
    input_features = processor(
        audio_flat,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features
    
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    print(f"Transcription: {transcription}")
    asyncio.run(stream_and_speak(transcription))


def start_space_bar_transcription():
    """Start the main transcription loop"""
    global stop_recording, is_recording
    
    stop_recording = False
    is_recording = False
    
    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype, callback=audio_callback):
        monitor_thread = threading.Thread(target=key_monitor)
        monitor_thread.start()
        
        try:
            while not stop_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_recording = True
            is_recording = False
            print("Interrupted by user")
        
        monitor_thread.join()
    
    print("Transcription stopped")                     

start_space_bar_transcription()

