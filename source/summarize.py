from transformers import pipeline

# Load the summarization pipeline
summarization_pipeline = pipeline(task="summarization", model="PaceKW/24PDInsight-TextSummarization")

def summarize_text(input_text):
    # Generate summary
    summarized_text = summarization_pipeline(input_text)[0]['summary_text']
    return summarized_text
