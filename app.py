from flask import Flask, request, jsonify
from source.transcribe import transcribe_youtube_audio, transcribe_mp3_audio
from source.wordcloud import wordcloud
import re

app = Flask(__name__)

# Regular expression to identify YouTube URLs
youtube_regex = re.compile(
    r'^(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.+$'
)

# Regular expression to identify valid MP3 URLs from Google Cloud Storage
mp3_regex = re.compile(
    r'^https://storage\.googleapis\.com/.+\.mp3$'
)

# Homepage
@app.route('/')
def index():
    return "Homepage"

# Transcribe
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # PASS FROM THE HEADER
        # Request the url link that pass from the backend
        # url = request.args.get('url')
        
        # For Test URL from BUCKET
        # url = 'https://storage.googleapis.com/files-bucket-24pdinsight/gibran.mp3'
        
        # For Test URL from YOUTUBE
        # url = 'https://youtu.be/OdptPKaEMFQ?si=nf8e1Jyq9bBLfpL7'
        
        # PASS FROM BODY
        # Request URL from user
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        if youtube_regex.match(url):
            # Transcribe audio from YouTube URL
            transcription = transcribe_youtube_audio(url)
        elif mp3_regex.match(url):
            # Transcribe audio from MP3 URL
            transcription = transcribe_mp3_audio(url)
        else:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Return the transcription result
        return jsonify({'transcription': transcription})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Wordcloud
@app.route('/wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        transcription = data.get('transcription')

        if not transcription:
            return jsonify({'error': 'No transcription provided'}), 400

        wordcloud_json = wordcloud(transcription)

        return wordcloud_json
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
