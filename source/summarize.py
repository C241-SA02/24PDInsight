from transformers import pipeline

# Load the summarization pipeline
summarization_pipeline = pipeline(task="summarization", model="PaceKW/24PDInsight-TextSummarization")

def summarize_text(transcription):
    # Generate summary
    summarized_text = summarization_pipeline(transcription)
    
    # Print the summarized text for debugging
    print("Summarized Text:", summarized_text)
    
    return summarized_text[0]['summary_text']
