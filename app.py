from flask import Flask, jsonify, request
from source.transcribe import transcribe_youtube_audio
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "HomePage"

@app.route('/transcribe', methods=['GET'])
def transcribe():
    try:
        # Get the YouTube URL from the query parameters
        youtube_url = request.args.get('url')
        
        # Use a default YouTube URL if none provided (for testing)
        if not youtube_url:
            youtube_url = 'https://youtu.be/OdptPKaEMFQ'
        
        transcription = transcribe_youtube_audio(youtube_url)
        
        # Return the transcription result
        return jsonify({'transcription': transcription})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
