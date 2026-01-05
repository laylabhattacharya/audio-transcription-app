# Import necessary libraries for audio processing and speech recognition
import sounddevice as sd  # Library for recording audio from microphone
import numpy as np  # Library for numerical operations on audio data arrays
import whisper  # OpenAI's Whisper model for speech-to-text transcription
from flask import Flask, request, jsonify, send_from_directory
import io
import wave

# Define a function to detect pauses in speech based on Whisper's timestamp data
# This function looks for gaps between spoken segments longer than the threshold
def check_pauses(segments, pause_threshold=0.5):
    # Initialize an empty list to store detected pauses
    pauses = []
    # Initialize an empty list to store unexpected commas (commas in text without corresponding pauses)
    commas = []
    # Loop through each segment starting from the second one
    for i in range(1, len(segments)):
        # Calculate the time gap between the end of the previous segment and start of current
        gap = segments[i]['start'] - segments[i-1]['end']
        # If the gap is longer than our threshold (0.5 second), consider it a pause
        if gap > pause_threshold:
            # Store the pause information: end time of previous, start of next, and gap duration
            pauses.append((segments[i-1]['end'], segments[i]['start'], gap))
    # Check each segment for commas in the text
    for segment in segments:
        if ',' in segment['text']:
            # Store the comma information: text snippet and start time
            commas.append((segment['text'].strip(), segment['start']))
    # Return the list of detected pauses and commas
    return pauses, commas

# Define a function to detect filler words in the transcribed text
# Filler words are common verbal crutches like "um", "like", etc.
def check_fillers(text):
    # Define a list of common filler words and phrases to look for
    fillers = ["um", "uh", "like", "kinda", "sort of", "you know", "i mean", "er", "ah", "so", "well", "actually"]
    # Initialize an empty list to store found fillers and their counts
    found_fillers = []
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    # Check each filler word in our list
    for filler in fillers:
        # If the filler word appears in the text
        if filler in text_lower:
            # Count how many times it appears
            count = text_lower.count(filler)
            # Add the filler and its count to our results list
            found_fillers.append((filler, count))
    # Return the list of found fillers with their counts
    return found_fillers

# Load the Whisper AI model for speech recognition
# Using 'small.en' model for better accuracy on English speech
print("Loading Whisper model...")
try:
    model = whisper.load_model("small.en")
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Failed to load Whisper model: {e}")
    model = None

# Initialize Flask app
app = Flask(__name__)

@app.route('/test')
def test_model():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Create a simple test audio signal (1 second of 440Hz sine wave)
    test_audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    
    try:
        result = model.transcribe(test_audio, fp16=False)
        return jsonify({
            'model_loaded': True,
            'test_transcription': result['text'].strip(),
            'test_segments': len(result['segments'])
        })
    except Exception as e:
        return jsonify({'error': f'Model test failed: {str(e)}'}), 500

# Set audio recording parameters
fs = 16000  # Sample rate: 16,000 samples per second (optimal for speech recognition)
duration = 30  # Recording duration in seconds

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if model is None:
        return jsonify({'error': 'Whisper model not loaded'}), 500
        
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    audio_data = audio_file.read()
    
    print(f"Received audio data: {len(audio_data)} bytes")
    print(f"Audio file name: {audio_file.filename}")
    print(f"Audio content type: {audio_file.content_type}")
    
    # Convert audio data to numpy array
    # The browser sends audio as a blob, try different decoding approaches
    audio_np = None
    
    # First, try raw PCM decoding (most likely for Web Audio API)
    try:
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"Raw PCM decoding successful, shape: {audio_np.shape}")
    except Exception as e:
        print(f"Raw PCM decoding failed: {e}")
    
    # If raw PCM fails, try WAV decoding
    if audio_np is None:
        try:
            import wave
            import io
            wav_buffer = io.BytesIO(audio_data)
            with wave.open(wav_buffer, 'rb') as wav_file:
                audio_np = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            print("WAV decoding successful")
        except Exception as e:
            print(f"WAV decoding failed: {e}")
    
    # If all else fails, return error
    if audio_np is None:
        return jsonify({'error': 'Could not decode audio data'}), 400
    
    print(f"Audio numpy array shape: {audio_np.shape}")
    print(f"Audio numpy array dtype: {audio_np.dtype}")
    print(f"Audio numpy array min/max: {audio_np.min():.6f} / {audio_np.max():.6f}")
    print(f"Audio duration estimate: {len(audio_np) / fs:.2f} seconds")
    
    # Validate audio data
    if len(audio_np) == 0:
        return jsonify({'error': 'Audio data is empty'}), 400
    
    if audio_np.max() - audio_np.min() < 0.001:
        print("WARNING: Audio appears to be silence or very quiet")
        return jsonify({'error': 'Audio appears to be silence or too quiet to transcribe'}), 400
    
    # Ensure audio is mono and at correct sample rate
    if len(audio_np.shape) > 1:
        print(f"Converting stereo to mono, original shape: {audio_np.shape}")
        audio_np = np.mean(audio_np, axis=1)  # Convert to mono
    
    # Check if we need to resample (Whisper expects 16kHz)
    # For now, assume browser audio is already at 16kHz, but add logging
    print(f"Final audio shape: {audio_np.shape}, sample rate assumed: {fs}Hz")
    
    # Additional validation: check for actual audio content
    # Calculate RMS to detect if there's meaningful audio
    rms = np.sqrt(np.mean(audio_np**2))
    print(f"Audio RMS level: {rms:.6f}")
    
    if rms < 0.001:
        print("WARNING: Audio RMS is very low, likely silence")
        return jsonify({'error': 'Audio appears to contain no speech or is too quiet'}), 400
    
    # Use Whisper to transcribe the audio to text
    print("Starting Whisper transcription...")
    text = ""
    segments = []
    
    try:
        result = model.transcribe(audio_np, fp16=False, without_timestamps=False)
        text = result["text"].strip()
        segments = result["segments"]
        
        print(f"Whisper transcription completed")
        print(f"Raw result text: '{result['text']}'")
        print(f"Stripped text: '{text}'")
        print(f"Number of segments: {len(segments)}")
        
        if len(segments) > 0:
            print(f"First segment: {segments[0]}")
            
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
    
    # If transcription is empty, try with different parameters
    if not text:
        print("Transcription was empty, trying with different parameters...")
        try:
            # Try without timestamps
            result2 = model.transcribe(audio_np, fp16=False, without_timestamps=True)
            text2 = result2["text"].strip()
            if text2:
                print(f"Alternative transcription successful: '{text2}'")
                text = text2
                segments = []  # No segments without timestamps
        except Exception as e2:
            print(f"Alternative transcription also failed: {e2}")
    
    # Final check
    if not text:
        print("All transcription attempts resulted in empty text")
        return jsonify({'error': 'No speech detected in audio'}), 400
    
    # Analyze pauses and commas
    pauses, commas = check_pauses(segments)
    
    # Analyze filler words
    fillers = check_fillers(text)
    
    # Prepare response
    response = {
        'transcription': text,
        'pauses': [{'end_prev': end_prev, 'start_next': start_next, 'duration': duration} for end_prev, start_next, duration in pauses],
        'commas': [{'text': text_snippet, 'start_time': start_time} for text_snippet, start_time in commas],
        'fillers': [{'word': filler, 'count': count} for filler, count in fillers]
    }
    
    return jsonify(response)

def main_cli():
    # Inform the user that we're calibrating for background noise
    print("Adjusting for ambient noise... Please be quiet.")
    # Record a 1-second sample of background noise (optional, Whisper handles this well)
    noise_sample = sd.rec(int(1 * fs), samplerate=fs, channels=1, dtype='int16')
    # Wait for the noise recording to complete
    sd.wait()

    # Prompt the user to speak and start the main recording
    print(f"Say something! Recording for {duration} seconds...")
    # Record audio from the microphone for the specified duration
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    # Wait for the recording to finish
    sd.wait()
    # Inform the user that recording is complete and transcription is starting
    print("Recording finished. Transcribing...")

    # Prepare the audio data for Whisper
    # Convert from 2D array to 1D, change data type to float32, and normalize values to -1 to 1 range
    audio_data = audio_data.flatten().astype(np.float32) / 32768.0

    # Use Whisper to transcribe the audio to text, requesting timestamp information for pause analysis
    result = model.transcribe(audio_data, fp16=False, without_timestamps=False)
    # Extract the transcribed text and remove any leading/trailing whitespace
    text = result["text"].strip()
    # Extract the timestamp segments for pause analysis
    segments = result["segments"]

    # Display the transcribed text to the user
    print("Transcription:", text)

    # Analyze the transcription for pauses using our custom function
    pauses, commas = check_pauses(segments)
    # If pauses were detected, display them
    if pauses:
        print("\nDetected pauses:")
        # Loop through each detected pause and print its details
        for end_prev, start_next, duration in pauses:
            print(f"  Pause from {end_prev:.2f}s to {start_next:.2f}s (duration: {duration:.2f}s)")
    # If no pauses detected, inform the user
    else:
        print("\nNo significant pauses detected.")

    # If commas were detected, display them
    if commas:
        print("\nDetected commas (potential short pauses):")
        # Loop through each detected comma and print its details
        for text_snippet, start_time in commas:
            print(f"  Comma in: '{text_snippet}' at {start_time:.2f}s")
    # If no commas detected, inform the user
    else:
        print("\nNo commas detected.")

    # Analyze the transcription for filler words using our custom function
    fillers = check_fillers(text)
    # If filler words were found, display them
    if fillers:
        print("\nDetected filler words:")
        # Loop through each found filler and print its count
        for filler, count in fillers:
            print(f"  '{filler}': {count} time(s)")
    # If no fillers detected, inform the user
    else:
        print("\nNo filler words detected.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        main_cli()
    else:
        app.run(debug=True)