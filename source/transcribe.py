from pytube import YouTube
from transformers import pipeline
import os
import requests

# Initialize the ASR pipeline once to avoid reloading the model on every request
whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium')

def transcribe(file_path):
    try:
        # Perform transcription on the downloaded audio file
        transcription = whisper(file_path)
        
        # Clean up the temporary file
        os.remove(file_path)
        
        # Return the transcription text
        return transcription['text']
    
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {str(e)}")

def transcribe_youtube_audio(youtube_url):
    try:
        # Download the audio stream from YouTube
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        # Download the audio stream to a temporary file
        temp_path = './temp_audio.mp4'
        audio_stream.download(output_path=os.path.dirname(temp_path), filename=os.path.basename(temp_path))
        
        # Transcribe the downloaded audio file
        return transcribe(temp_path)
    
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {str(e)}")

def transcribe_mp3_audio(mp3_url):
    try:
        # Download the MP3 file from the provided URL
        temp_path = './temp_audio.mp3'
        response = requests.get(mp3_url)
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # Transcribe the downloaded MP3 file
        return transcribe(temp_path)
    
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {str(e)}")