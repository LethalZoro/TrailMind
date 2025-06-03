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
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import time

# Load the model and processor
processor_image = AutoImageProcessor.from_pretrained("dima806/medicinal_plants_image_detection")
model_image = AutoModelForImageClassification.from_pretrained("dima806/medicinal_plants_image_detection")


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


def classify_medicinal_plant(image_path):
    # Load the local image file
    image = Image.open(image_path)
    
    # Process the image for model input
    inputs = processor_image(images=image, return_tensors="pt")
    
    # Time the prediction
    time_start = time.time()
    # Get model predictions
    outputs = model_image(**inputs)
    time_end = time.time()
    
    # Calculate prediction time
    prediction_time = time_end - time_start
    print(f"Time taken for prediction: {prediction_time:.4f} seconds")
    
    # Process predictions
    predictions = outputs.logits.softmax(dim=1)
    
    # Get the predicted class
    predicted_class_idx = predictions.argmax().item()
    predicted_class = model_image.config.id2label[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx].item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    return {
        "class": predicted_class,
        "confidence": confidence,
        "prediction_time": prediction_time
    }

def capture_and_classify_plant():
    """
    Captures an image using the camera, classifies it using the model, 
    and returns the predicted class with its confidence.
    """
    from picamera2 import Picamera2, Preview
    import time
    from PIL import Image

    # Initialize the camera
    picam2 = Picamera2()
    # picam2.start_preview(Preview.QTGL)
    preview_config = picam2.create_preview_configuration()
    capture_config = picam2.create_still_configuration()

    # Configure and start the camera
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)  # Allow the camera to stabilize

    # Capture the image
    image = picam2.switch_mode_and_capture_image(capture_config)
    picam2.close()

    # Save the image temporarily
    temp_image_path = "captured_image.jpg"
    image.save(temp_image_path)

    # Classify the captured image
    result = classify_medicinal_plant(temp_image_path)

    # Return the result
    return result

# Main function: stream from Ollama, buffer sentences, and queue them
async def stream_and_speak(query):
    queue = asyncio.Queue()
    speaker_task = asyncio.create_task(speak_from_queue(queue))
    image_task = False
    if re.search(r"\bwhat\s+is\s+this\b", query, re.IGNORECASE):
        image_task = True

    if image_task:
        result=capture_and_classify_plant()
    else:
        result = {"class": "unknown", "confidence": 0.0, "prediction_time": 0.0}

    prompt_normal=(
            f"You are an outdoor voice assistant. Be precise in your answers.\n"
            f"Current query: {query}\n"
            f"Remember to answer the question without any preamble or introduction. Be to the point.\n"
            f"Answer in a single sentence, no more than 50 words.\n"
        )
    prompt_image=(
            f"You are an outdoor voice assistant. Be precise in your answers.\n"
            f"The user has captured an image of a plant."
            f"the classification result is: {result['class']} with confidence {result['confidence']:.2%}.\n"
            f"Briefly describe the plant and its uses in a single sentence, no more than 50 words.\n"
        )
    response = ollama.generate(
        model="smollm:135m",
        prompt= prompt_image if image_task else prompt_normal,
        stream=True,
        options={"temperature": 0.7, "num_predict": 50}
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

# def key_monitor():
#     """Monitor space bar and ESC using pynput"""
#     global is_recording, stop_recording

#     def on_press(key):
#         global is_recording
#         try:
#             # Check if key is the space key using pynput's Key.space
#             if key == pynput_keyboard.Key.space and not is_recording:
#                 is_recording = True
#                 audio_queue.queue.clear()
#                 print("Recording... (holding space)")
#         except AttributeError:
#             pass

#     def on_release(key):
#         global is_recording, stop_recording
#         try:
#             if key == pynput_keyboard.Key.space and is_recording:
#                 is_recording = False
#                 print("Processing...")
#                 process_audio()
#         except AttributeError:
#             if key == pynput_keyboard.Key.esc:
#                 stop_recording = True
#                 is_recording = False
#                 print("Stopping...")
#                 return False  # Stop listener

#     print("Press and hold SPACE to record. Release to transcribe. Press ESC to quit.")
#     with pynput_keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#         listener.join()

def key_monitor():
    """Monitor space bar and ESC using direct terminal input"""
    global is_recording, stop_recording
    
    import termios
    import tty
    import sys
    import select
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        print("\rPress and hold SPACE to record. Release to transcribe. Press ESC or Q to quit.")
        
        # Track space bar state
        space_pressed = False
        
        while not stop_recording:
            # Check if input is available (non-blocking)
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            
            if r:
                ch = sys.stdin.read(1)
                
                # Space pressed
                if ch == ' ':
                    if not space_pressed:
                        space_pressed = True
                        is_recording = True
                        audio_queue.queue.clear()
                        print("\rRecording... (holding space)")
                # Space released (any other key after space was pressed)
                elif space_pressed:
                    space_pressed = False
                    is_recording = False
                    print("\rProcessing...")
                    process_audio()
                    print("\rPress and hold SPACE to record. Press ESC or Q to quit.")
                
                # ESC or q to quit
                if ch == '\x1b' or ch.lower() == 'q':
                    stop_recording = True
                    is_recording = False
                    print("\rStopping...")
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nKeyboard monitor stopped")
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

