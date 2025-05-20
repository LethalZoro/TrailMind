# en_US-kathleen-low.onnx

import subprocess

def text_to_speech(text):
    # Define the command to execute
    command = f"echo '{text}' | ./piper/piper/piper --model ./piper/en_US-kathleen-low.onnx --rate 125 --output-raw | aplay -r 16050 -f S16_LE -t raw -"

    # Execute the command
    subprocess.run(command, shell=True)
text = "This sentence is spoken first. This sentence is synthesized while the first sentence is spoken."
text_to_speech(text)
