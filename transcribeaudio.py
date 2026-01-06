
import sounddevice as sd
import numpy as np
import whisper
from flask import Flask, request, jsonify, send_from_directory
import io
import wave
import ffmpeg
import tempfile
import os
# Import analysis functions from audio_analysis.py
from audio_analysis import check_pauses, check_fillers, check_repeats


app = Flask(__name__)

# Load Whisper model at startup
print("Loading Whisper model...")
try:
    model = whisper.load_model("small.en")
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Failed to load Whisper model: {e}")
    model = None

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

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
    
    # Use ffmpeg-python to handle various audio formats
    try:
        # Set ffmpeg path explicitly (winget installs to this location)
        ffmpeg_path = r"C:\Users\layla\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
        
        # Write audio data to temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Use ffmpeg to convert to WAV format that we can easily read
            wav_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            wav_temp.close()
            
            # Convert WebM to WAV using ffmpeg with explicit path
            stream = ffmpeg.input(temp_file_path)
            stream = ffmpeg.output(stream, wav_temp.name, acodec='pcm_s16le', ac=1, ar=16000)
            ffmpeg.run(stream, cmd=ffmpeg_path, overwrite_output=True, quiet=True)
            
            # Read the WAV file
            import wave
            with wave.open(wav_temp.name, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            print(f"Audio loaded and converted successfully: {len(audio_np)} samples, 16000Hz, mono")
            print(f"Audio converted to numpy array, shape: {audio_np.shape}")
            
            # Clean up WAV temp file
            os.unlink(wav_temp.name)
            
        finally:
            # Clean up WebM temp file
            os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"Audio decoding failed: {e}")
        return jsonify({'error': f'Could not decode audio data: {str(e)}'}), 400
    
    print(f"Audio numpy array shape: {audio_np.shape}")
    print(f"Audio numpy array dtype: {audio_np.dtype}")
    print(f"Audio numpy array min/max: {audio_np.min():.6f} / {audio_np.max():.6f}")
    print(f"Audio duration estimate: {len(audio_np) / 16000:.2f} seconds")
    
    # Validate audio data
    if len(audio_np) == 0:
        return jsonify({'error': 'Audio data is empty'}), 400
    
    if audio_np.max() - audio_np.min() < 0.001:
        print("WARNING: Audio appears to be silence or very quiet")
        return jsonify({'error': 'Audio appears to be silence or too quiet to transcribe'}), 400
    
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
    
    # Detect repeated words/phrases
    repeated_words, repeated_phrases = check_repeats(text)
    # Prepare response
    response = {
        'transcription': text,
        'pauses': [{'end_prev': end_prev, 'start_next': start_next, 'duration': duration} for end_prev, start_next, duration in pauses],
        'commas': [{'text': text_snippet, 'start_time': start_time} for text_snippet, start_time in commas],
        'fillers': [{'word': filler, 'count': count} for filler, count in fillers],
        'repeated_words': repeated_words,
        'repeated_phrases': repeated_phrases
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