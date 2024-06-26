import os
import json
import requests
from pytube import YouTube
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv('API_KEY')
if api_key is None:
    raise ValueError("OpenAI API key not found in the environment. Make sure to set it in the .env file.")

client = OpenAI(api_key=api_key)

def transcribe(file_path):
    try:
        # Perform transcription on the downloaded audio file using OpenAI API
        with open(file_path, "rb") as audio_data:
            transcription = client.audio.transcriptions.create(
                model='whisper-1',
                file=audio_data,
                language='id')

        # Clean up the temporary file
        os.remove(file_path)

        # Return the transcription text as JSON
        return transcription.text

    except Exception as e:
        raise RuntimeError(f"Error during transcription: {str(e)}")

def transcribe_youtube_audio(youtube_url):
    try:
        # Download the audio stream from YouTube
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()

        # Download the audio stream to a temporary file
        temp_path = './temp_audio.mp3'
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

if __name__ == "__main__":
    url = "https://youtu.be/OdptPKaEMFQ?si=w7w_u6Zvp5bxpVug"
    transcript = transcribe_youtube_audio(url)
    print("Transcription:", transcript)
