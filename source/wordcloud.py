import json
from wordcloud import WordCloud

def wordcloud(transcription):
    try:
        # Generate word frequencies
        wordcloud = WordCloud().process_text(transcription)
        
        # Convert to desired format with word key before size key
        wordcloud_data = [{'word': word, 'size': str(frequency)} for word, frequency in wordcloud.items()]

        # Convert the list of dictionaries to JSON string
        wordcloud_json = json.dumps({'wordcloud': wordcloud_data}, sort_keys=False)

        return wordcloud_json

    except Exception as e:
        return json.dumps({'error': str(e)}), 500
