# Day 105: Capstone: Building a Multimodal Future
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing "Jarvis" (Simplified)

We will build a loop that takes an Image + Question -> Speaks the Answer.

```python
import cv2
from openai import OpenAI
import base64

client = OpenAI()

class Jarvis:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def see(self):
        ret, frame = self.cam.read()
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def hear(self):
        # (Assume Whisper handles this)
        return input("You: ")

    def speak(self, text):
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file("output.mp3")
        # Play audio...

    def think(self, image_b64, text):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content

    def run(self):
        while True:
            text = self.hear()
            if text == "exit": break
            
            print("Looking...")
            img = self.see()
            
            print("Thinking...")
            response = self.think(img, text)
            
            print(f"Jarvis: {response}")
            self.speak(response)

# Usage
# jarvis = Jarvis()
# jarvis.run()
```

### Advanced: Tool Use with Vision

"Jarvis, buy this shoe."
1.  **See:** Capture image of shoe.
2.  **Think:** Extract product details (Nike Air Max, Size 10).
3.  **Act:** Call `amazon_search(query="Nike Air Max")`.
4.  **Verify:** Compare search result image with camera image (Visual Similarity).
5.  **Buy:** Call `add_to_cart()`.

### Summary

*   **Multimodal Context:** The `messages` array in GPT-4o supports mixed Text/Image content.
*   **State:** Real-world agents need a State Machine (Listening -> Thinking -> Speaking -> Acting).
