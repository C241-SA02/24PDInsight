from flask import Flask, jsonify
from pytube import YouTube
from transformers import pipeline
import os

app = Flask(__name__)

# Initialize the ASR pipeline once to avoid reloading the model on every request
whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium')

def transcribe_youtube_audio(youtube_url):
    try:
        # Download the audio stream from YouTube
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        # Download the audio stream to a temporary file
        temp_path = './temp_audio.mp4'
        audio_stream.download(output_path=os.path.dirname(temp_path), filename=os.path.basename(temp_path))
        
        # Perform transcription on the downloaded audio file
        transcription = whisper(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        # Return the transcription text
        return transcription['text']
    
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {str(e)}")

@app.route('/transcribe', methods=['GET'])
def transcribe():
    try:
        # Call the transcription function with the provided YouTube URL
        youtube_url = 'https://youtu.be/OdptPKaEMFQ'
        transcription = transcribe_youtube_audio(youtube_url)
        
        # Return the transcription result
        return jsonify({'transcription': transcription})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
