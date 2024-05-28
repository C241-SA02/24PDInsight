from pytube import YouTube
from transformers import pipeline
import os

# Initialize the ASR pipeline once to avoid reloading the model on every request
whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium')

def transcribe_youtube_audio(youtube_url, temp_path='./temp_audio.mp4'):
    try:
        # Download the audio stream from YouTube
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        # Download the audio stream to the temporary path
        audio_stream.download(output_path=os.path.dirname(temp_path), filename=os.path.basename(temp_path))
        
        # Perform transcription on the downloaded audio file
        transcription = whisper(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        # Return the transcription result
        return transcription[0]['transcription']
    
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {str(e)}")
