import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Set seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Load model and tokenizer from Hugging Face
model_name = "farizkuy/SA_fine_tuned"
sentiment_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define label for sentiment analysis
id2label = {0: "positive", 1: "neutral", 2: "negative"}

def analyze_sentiment(text):
    # Tokenize and analyze the text
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    outputs = sentiment_model(inputs)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    # Prepare output in the desired format
    output_list = [{"label": label, "score": f"{prob:.3f}"} for label, prob in zip(id2label.values(), probabilities)]
    return output_list
