from flask import Flask, request, jsonify
from source.transcribe import transcribe_youtube_audio

app = Flask(__name__)

# Index route
@app.route('/', methods=['GET'])
def index():
    return 'Welcome to the transcription service!'

@app.route('/transcribe', methods=['GET'])
def transcribe():
    try:
        # Get the YouTube URL from the query parameters
        # youtube_url = request.args.get('url')
        
        # Test
        youtube_url = 'https://youtu.be/OdptPKaEMFQ?si=YLhFMkzW_lukpiWC'
        
        # # Check if youtube_url is None
        # if youtube_url is None:
        #     # Use a default YouTube URL if none provided
        #     youtube_url = 'https://youtu.be/OdptPKaEMFQ'
        
        # Call the transcription function
        transcription = transcribe_youtube_audio(youtube_url)
        
        # Return the transcription result
        return jsonify({'transcription': transcription})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
