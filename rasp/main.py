import speech_recognition as sr
from gtts import gTTS
import subprocess
import os
from openai import OpenAI
import pyttsx3
import asyncio
import string
import numpy as np
import time
# engine = pyttsx3.init()
# engine.setProperty('rate', 125)
# voices = engine.getProperty('voices')
# # print(voices)
# engine.setProperty('voice', voices[0].id)

# volume = engine.getProperty('volume')
# # print (volume)
# engine.setProperty('volume',0.7)

client = OpenAI(
    api_key="")
# ASSISTANT_ID ="asst_fWfbAUgTBe0FbftmfNGXRueb"
ASSISTANT_ID = "asst_fWfbAUgTBe0FbftmfNGXRueb"

thread = client.beta.threads.create()

name = "zoro"
wake_up_words = ["hey", "wake up", "hello", "hi", "what's up"]
stop_commands = ["exit", "stop", "quit", "goodbye",
                 "shutdown", "shutoff", "power off", "power down"]
greetings = [f"whats up master {name}",
             "yeah?",
             "Well, hello there, Master of Puns and Jokes - hows it going today?",
             f"Ahoy there, Captain {name}! How's the ship sailing?",
             f"Bonjour, Monsieur {name}! Comment  va vola? Wait, why the hell am I speaking French?"]

recognizer = sr.Recognizer()


def listen():

    i = 1
    while i:
        i = i-1
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)

            audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text

        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")

            asyncio.run(
                speak("Sorry, could not understand audio. Please speak again"))
            i = i+1
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}")

            asyncio.run(speak("Error in Speech Recognition. Please try again"))
            exit()


def chat_bot(message):
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )
    print(f"Run created: {run.id}")
    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        print(f"Run status: {run.status}")
    else:
        print(f"Run status: {run.status}")
    if run.status == 'completed':
        chat_history = client.beta.threads.messages.list(
            thread_id=thread.id
        )
    latest_message = chat_history.data[0]
    # print(latest_message.content[0].text.value)
    response = latest_message.content[0].text.value
    return response


async def speak(text):
    # engine.say(text)
    # engine.runAndWait()
    command = f"echo '{text}' | ./piper/piper/piper --model ./piper/en_US-kathleen-low.onnx --rate 125 --output-raw | aplay -r 16050 -f S16_LE -t raw -"

    # Execute the command
    text_length = len(text)
    duration_per_character = 0.05  # Example: 50 milliseconds per character
    estimated_duration = text_length * duration_per_character

    subprocess.run(command, shell=True)
    await asyncio.sleep(1)
    print("TTS is done")


def wake_up():

    while True:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            temp_u = text.lower().split()
            temp_u = temp_u[:5]
            if (any(word in wake_up_words for word in temp_u)):
                print("Waking up voice chat bot.")
                asyncio.run(speak(np.random.choice(greetings)))
                return text

            if (any(word in stop_commands for word in temp_u)):
                print("Exiting voice chat bot.")
                asyncio.run(speak("Goodbye"))
                # close the program
                exit()
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
            pass
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}")
            exit()


def voice_chat_bot():

    wake_up()
    while True:
        user_input = listen()
        temp_u = user_input.lower().split()
        temp_u = temp_u[:5]
        if (any(word in stop_commands for word in temp_u)):
            print("Exiting voice chat bot.")
            return

        hf_response = chat_bot(user_input)

        output = str(hf_response)
        start_index = output.find('bot >>') + len('bot >>')
        bot_text = output[start_index:].strip()
        asyncio.run(speak(bot_text))
    return bot_text


res = voice_chat_bot()
