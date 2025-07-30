# Real-time ASL Recognition

A real-time American Sign Language (ASL) recognition system using computer vision and deep learning. The system can recognize both individual ASL alphabets and complete words/sentences.

## Features

- Real-time hand and pose detection using MediaPipe
- Two recognition modes:
  - MLP mode for recognizing ASL alphabets
  - LSTM mode for recognizing complete words
- Seq2seq model for sentence processing
- Live video feed with visualization of hand landmarks
- FPS counter and confidence display

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python app.py
```

### Controls

- Press 'q' to switch between MLP (alphabet) and LSTM (word) modes
- Press 'ESC' to exit the application

### Recognition Modes

1. **MLP Mode (Alphabet Recognition)**

   - Recognizes individual ASL alphabets (A-Z, except J)
   - Shows real-time confidence scores
   - Adds letters to the sentence when hands are removed from view

2. **LSTM Mode (Word Recognition)**
   - Recognizes complete words from ASL gestures
   - Supported words: "hello", "meet", "my", "name", "nice", "please", "sit", "yes", "you"
   - Processes sequences of hand movements
   - Adds words to the sentence when gesture is completed

## Project Structure

- `app.py`: Main application file
- `mlp/`: MLP model for alphabet recognition
- `lstm/`: LSTM model for word recognition
- `seq2seq/`: Sequence-to-sequence model for sentence processing
