# Day 13.2: Advanced RNN Applications - A Practical Guide

## Introduction: The Versatility of Recurrent Architectures

While machine translation (Seq2Seq) is a classic application, the power of RNNs, LSTMs, and GRUs extends to a vast array of other creative and practical tasks. By understanding how to shape the inputs and interpret the outputs, we can adapt these recurrent architectures to solve diverse problems.

This guide provides a practical overview of several advanced and interesting applications of RNNs, complete with conceptual explanations and PyTorch implementation sketches.

**Today's Learning Objectives:**

1.  **Perform Sentiment Analysis with RNNs:** Revisit this classic task with a focus on how to handle the variable-length output of an RNN for a single classification.
2.  **Explore Music Generation:** Understand how an RNN can be trained to predict the next note in a sequence, allowing it to generate simple melodies.
3.  **Understand Time Series Anomaly Detection:** See how a trained forecasting model can be used as an anomaly detector by measuring the error between its predictions and the actual data.
4.  **Grasp the Concept of Image Captioning:** Learn about the hybrid CNN-RNN architecture used to generate descriptive text captions for images.

---

## Part 1: Sentiment Analysis Revisited

**The Task:** Classify a piece of text (e.g., a movie review) as having positive or negative sentiment.

**The Architecture:** This is a **many-to-one** task. The input is a sequence of words, but the output is a single classification.

**The Key Idea:**
1.  The input sentence is passed through an embedding layer.
2.  The sequence of embeddings is fed into an LSTM or GRU.
3.  The RNN processes the entire sequence. We are interested in the final summary of the sentence, which is captured in the **final hidden state** of the RNN.
4.  This final hidden state vector is then passed through a standard linear layer to produce the final classification (e.g., a single logit for a binary positive/negative decision).

### 1.1. Implementation Sketch

```python
import torch
import torch.nn as nn

print("--- Part 1: Sentiment Analysis ---")

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        # Pack sequence to handle padding
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        
        # We don't need the `output` of the LSTM, only the final hidden and cell states.
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        # Concatenate the final forward and backward hidden states of the last layer
        # hidden shape: [num_layers * num_directions, batch, hid_dim]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        return self.fc(hidden)

# --- Dummy Usage ---
model = SentimentClassifier(vocab_size=10000, embedding_dim=100, hidden_dim=256, output_dim=1, 
                            n_layers=2, bidirectional=True, dropout=0.5, pad_idx=1)

# A dummy batch of text data
text_batch = torch.randint(0, 10000, (32, 50)) # (batch, seq_len)
text_lengths = torch.full((32,), 50) # Assume all are full length for simplicity

prediction = model(text_batch, text_lengths)
print(f"Input shape: {text_batch.shape}")
print(f"Output prediction shape: {prediction.shape}") # (batch, output_dim)
```

---

## Part 2: Music Generation

**The Task:** Generate a new sequence of musical notes that sounds coherent.

**The Architecture:** This is a **many-to-many** task, similar to character-level language modeling. The model learns to predict the *next* note in a sequence given all the previous notes.

**The Key Idea:**
1.  Represent musical notes numerically (e.g., MIDI note numbers).
2.  Create input/output pairs where the input is a sequence of notes and the target is the same sequence shifted by one time step.
    *   Input: `[note_1, note_2, ..., note_{n-1}]`
    *   Target: `[note_2, note_3, ..., note_n]`
3.  Train an LSTM to take a sequence and predict the output for *every* time step.
4.  For generation, provide a seed sequence, get the prediction for the next note, append that note to the sequence, and feed it back into the model to generate the next note, and so on.

### 2.1. Implementation Sketch

```python
print("\n--- Part 2: Music Generation ---")

class MusicGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        # vocab_size here is the number of possible notes
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden_and_cell=None):
        embedded = self.embedding(x)
        # The LSTM output contains the hidden state for every time step
        output, (h, c) = self.lstm(embedded, hidden_and_cell)
        # We pass every time step's output through the linear layer
        predictions = self.fc(output)
        return predictions, (h, c)

# --- Dummy Usage ---
model = MusicGenerator(vocab_size=88, embedding_dim=64, hidden_dim=256, n_layers=2)

# A dummy batch of musical sequences
music_seq_batch = torch.randint(0, 88, (32, 100)) # (batch, seq_len)

# Get predictions for the next note at every time step
predictions, _ = model(music_seq_batch)
print(f"Input shape: {music_seq_batch.shape}")
print(f"Output predictions shape: {predictions.shape}") # (batch, seq_len, vocab_size)
```

---

## Part 3: Time Series Anomaly Detection

**The Task:** Given a time series of sensor data, identify time points that are anomalous or represent faulty behavior.

**The Architecture:** This is an unsupervised or semi-supervised task that cleverly uses a forecasting model.

**The Key Idea:**
1.  Train a high-performing LSTM forecasting model on a large amount of **normal, non-anomalous** time series data.
2.  Once trained, use this model in production. At each time step, feed it the recent history and ask it to predict the next value.
3.  Calculate the **reconstruction error**: the difference (e.g., Mean Absolute Error) between the model's prediction and the actual value that just occurred.
4.  If the reconstruction error exceeds a pre-defined **threshold**, flag that time point as an **anomaly**. The logic is that the model knows how normal data should behave, so when it sees something unexpected, its prediction will be poor, leading to a large error.

### 3.1. Implementation Sketch

```python
print("\n--- Part 3: Time Series Anomaly Detection ---")

# We would use a trained forecasting model like the one from Day 12.4
# For this sketch, let's assume `forecasting_model` is a pre-trained LSTM.
forecasting_model = nn.LSTM(1, 50, batch_first=True) # Dummy model
forecasting_model.eval()

def detect_anomalies(model, data, window_size, threshold):
    anomalies = []
    for i in range(len(data) - window_size):
        window = data[i:i+window_size].view(1, window_size, 1)
        actual_next_val = data[i+window_size]
        
        with torch.no_grad():
            predicted_next_val = model(window)[0][:, -1, :].item() # Simplified
        
        reconstruction_error = abs(actual_next_val - predicted_next_val)
        
        if reconstruction_error > threshold:
            print(f"Anomaly detected at time step {i+window_size}! Error: {reconstruction_error:.4f}")
            anomalies.append(i+window_size)
    return anomalies

# --- Dummy Usage ---
dummy_normal_data = torch.sin(torch.linspace(0, 100, 200)) 
dummy_normal_data[150] += 1.5 # Introduce a large spike anomaly

# In a real scenario, the threshold would be determined from a validation set.
anomaly_threshold = 0.5 
detect_anomalies(forecasting_model, dummy_normal_data, 12, anomaly_threshold)
```

---

## Part 4: Image Captioning

**The Task:** Generate a human-like text description for a given image.

**The Architecture:** This is the classic **hybrid CNN-RNN** model, a beautiful example of combining different architectures.

**The Key Idea:**
1.  **Encoder (CNN):** The input image is passed through a pre-trained Convolutional Neural Network (like a ResNet or VGG), with the final classification layer removed. The output is a rich feature vector that represents the *content* of the image. This is the equivalent of the "context vector" in a Seq2Seq model.
2.  **Decoder (RNN/LSTM):** This feature vector is used as the **initial hidden state** of an LSTM decoder. The decoder then generates the caption word by word, just like the decoder in a Seq2Seq model. It takes a `<sos>` token, generates the first word, feeds that word back in to generate the second, and so on, until it produces an `<eos>` token.

### 4.1. Implementation Sketch

```python
print("\n--- Part 4: Image Captioning ---")

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        # --- Encoder Part ---
        # Load a pre-trained CNN and remove the final layer
        resnet = torchvision.models.resnet50(weights='DEFAULT')
        modules = list(resnet.children())[:-1] # Remove the final fc layer
        self.cnn_encoder = nn.Sequential(*modules)
        # A linear layer to map the CNN features to the LSTM's hidden size
        self.fc_init = nn.Linear(resnet.fc.in_features, hidden_size)
        
        # --- Decoder Part ---
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        # --- Encoder Pass ---
        with torch.no_grad(): # We don't train the pre-trained CNN
            features = self.cnn_encoder(images)
        features = features.view(features.size(0), -1)
        # Use the image features to initialize the LSTM's hidden state
        initial_hidden = self.fc_init(features).unsqueeze(0)
        initial_cell = torch.zeros_like(initial_hidden)
        
        # --- Decoder Pass ---
        embedded_captions = self.embedding(captions)
        outputs, _ = self.lstm(embedded_captions, (initial_hidden, initial_cell))
        predictions = self.fc_out(outputs)
        return predictions

# --- Dummy Usage ---
model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=10000)

images = torch.randn(8, 3, 224, 224) # Batch of 8 images
captions = torch.randint(0, 10000, (8, 15)) # Batch of 8 corresponding captions

output_logits = model(images, captions)
print(f"Input image shape: {images.shape}")
print(f"Input caption shape: {captions.shape}")
print(f"Output logits shape: {output_logits.shape}") # (batch, seq_len, vocab_size)
```

## Conclusion

Recurrent Neural Networks are not a one-trick pony. They are a flexible and powerful tool that can be adapted to a wide range of creative and practical applications beyond simple forecasting or translation.

**Key Architectural Patterns:**

*   **Many-to-One (e.g., Sentiment Analysis):** Process a full sequence and use the final hidden state for a single prediction.
*   **Many-to-Many (e.g., Music/Text Generation):** Process a sequence and predict the next item at each time step.
*   **Forecasting for Anomaly Detection:** Use a trained model's prediction error as a signal for anomalous behavior.
*   **Hybrid CNN-RNN (e.g., Image Captioning):** Use a CNN as a powerful feature extractor to create a context vector that can initialize the state of an RNN decoder.

By understanding these patterns, you can start to see how different deep learning architectures can be used as building blocks and combined to solve even the most complex multi-modal tasks.

## Self-Assessment Questions

1.  **Many-to-One:** In the sentiment analysis model, which specific tensor is used as the input to the final classification layer?
2.  **Generation:** When generating music or text, how is the output from time step `t` used to generate the output for time step `t+1`?
3.  **Anomaly Detection:** What is the "reconstruction error" in the context of time series anomaly detection?
4.  **Image Captioning:** What is the role of the CNN in an image captioning model? What is the role of the RNN?
5.  **Hybrid Architectures:** What information is passed from the CNN encoder to the RNN decoder in the image captioning model?

