# Lab 2: Voice Assistant (Jarvis)

## Objective
Build a voice loop: Speak -> Transcribe -> Think -> Speak.

## 1. The Loop (`jarvis.py`)

```python
from openai import OpenAI
client = OpenAI()

def listen():
    # Mocking microphone input
    return input("You (Microphone): ")

def think(text):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}]
    ).choices[0].message.content

def speak(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file("output.mp3")
    print(f"AI (Speaker): {text} [Saved to output.mp3]")

# Run
while True:
    user_input = listen()
    if user_input == "exit": break
    
    response = think(user_input)
    speak(response)
```

## 2. Challenge
*   **Real Microphone:** Use `speech_recognition` library to replace `input()`.
*   **Real Playback:** Use `playsound` library to play the MP3 automatically.

## 3. Submission
Submit the conversation log.
