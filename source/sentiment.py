import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

# Set seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Load model and tokenizer from Hugging Face
model_name = "farizkuy/SA_fine_tuned"
sentiment_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define label for sentiment analysis
id2label = {0: "positive", 1: "neutral", 2: "negative"}

def split_text_into_segments(text, tokenizer, max_length=512):
    tokens = tokenizer(text, return_tensors="tf", padding=False, truncation=False)["input_ids"][0]
    num_tokens = len(tokens)
    segments = []
    for i in range(0, num_tokens, max_length):
        segment = tokens[i:i + max_length]
        segments.append(segment)
    return segments

def predict_sentiment_for_segments(segments, model):
    predictions = []
    for segment in segments:
        inputs = {"input_ids": tf.expand_dims(segment, 0)}
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predictions.append(probabilities)
    return predictions

def analyze_sentiment(text):
    # Split text into segments
    segments = split_text_into_segments(text, tokenizer)
    
    # Predict sentiment for each segment
    predictions = predict_sentiment_for_segments(segments, sentiment_model)
    
    # Aggregate predictions
    average_probabilities = np.mean(predictions, axis=0)
    
    # Prepare output in the desired format
    output_list = [{"label": label, "score": f"{prob:.3f}"} for label, prob in zip(id2label.values(), average_probabilities)]
    return output_list
