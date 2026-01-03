# Audio Transcription App

This Python application transcribes speech from the microphone to text using OpenAI's Whisper model.

## Setup

1. Ensure you have Python installed.
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the script to start recording from the microphone:

`python transcribeaudio.py`

The script will record audio for 5 seconds and transcribe your speech to text using the Whisper model. It will also analyze the transcription for pauses (gaps longer than 1 second) and filler words (like "um", "uh", "like", etc.).

## Features

- **Speech Transcription**: Converts spoken words to text
- **Pause Detection**: Identifies significant pauses in speech (gaps > 1 second)
- **Filler Word Analysis**: Counts occurrences of common filler words

## Requirements

- openai-whisper library
- sounddevice library (for microphone access)
- scipy library
- torch (automatically installed with openai-whisper)
- Microphone

## Model Selection

The script uses the 'tiny' model by default for lower memory usage. You can change it to 'base', 'small', 'medium', or 'large' in the script for different accuracy/speed trade-offs.

## Troubleshooting

- Ensure your microphone is connected and working.
- The first run may take longer as the model downloads.
- The recording duration is fixed at 5 seconds; speak clearly within that window.
- Whisper works offline once the model is downloaded.