import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Initialize model and tokenizer for NER
ner_model_name = "farizkuy/ner_fine_tuned"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

# Translation dictionary for labels
label_translation = {
    'B-CRD': 'Angka', 'B-DAT': 'Tanggal', 'B-EVT': 'Peristiwa', 'B-FAC': 'Fasilitas', 'B-GPE': 'Entitas Geologi',
    'B-LAN': 'Bahasa', 'B-LAW': 'Hukum', 'B-LOC': 'Lokasi', 'B-MON': 'Uang', 'B-NOR': 'Norma',
    'B-ORD': 'Ordinat', 'B-ORG': 'Organisasi', 'B-PER': 'Orang', 'B-PRC': 'Proses', 'B-PRD': 'Produk',
    'B-QTY': 'Jumlah', 'B-REG': 'Agama', 'B-TIM': 'Waktu', 'B-WOA': 'Karya',
    'I-CRD': 'Angka', 'I-DAT': 'Tanggal', 'I-EVT': 'Peristiwa', 'I-FAC': 'Fasilitas', 'I-GPE': 'Entitas Geologi',
    'I-LAN': 'Bahasa', 'I-LAW': 'Hukum', 'I-LOC': 'Lokasi', 'I-MON': 'Uang', 'I-NOR': 'Norma',
    'I-ORD': 'Ordinat', 'I-ORG': 'Organisasi', 'I-PER': 'Orang', 'I-PRC': 'Proses', 'I-PRD': 'Produk',
    'I-QTY': 'Jumlah', 'I-REG': 'Agama', 'I-TIM': 'Waktu', 'I-WOA': 'Karya', 'O': 'O'
}

def ner_predict_and_translate(text, model, tokenizer, label_translation):
    # Tokenize input text
    tokens = text.split()
    tokenized_inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
    outputs = model(**tokenized_inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert tokens and predicted labels to words and labels
    words = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])
    labels = [model.config.id2label[label_id.item()] for label_id in predictions[0]]

    # Translate labels
    translated_labels = [label_translation[label] for label in labels]

    # Filter out special tokens and return the result
    result = [(word, label) for word, label in zip(words, translated_labels) if word not in tokenizer.all_special_tokens]
    return result

def ner_format_result(result):
    # Format the NER result for display
    formatted_text = ""
    for word, label in result:
        if label != 'O':
            formatted_text += f"{word} [{label}] "
        else:
            formatted_text += f"{word} "
    return formatted_text.strip()

def analyze_ner(text):
    # Analyze text for NER and format the result
    ner_result = ner_predict_and_translate(text, ner_model, ner_tokenizer, label_translation)
    formatted_ner_result = ner_format_result(ner_result)
    return formatted_ner_result

if __name__ == "__main__":
    # Sample text for testing
    sample_text = "Barack Obama was born on August 4, 1961 in Honolulu, Hawaii."
    
    # Expected format: [('Barack', 'Orang'), ('Obama', 'Orang'), ('was', 'O'), ('born', 'O'), ('on', 'O'), ('August', 'Tanggal'), ('4', 'Tanggal'), ('1961', 'Tanggal'), ('in', 'O'), ('Honolulu', 'Lokasi'), ('Hawaii', 'Lokasi'), ('.', 'O')]
    
    # Analyze the sample text
    result = analyze_ner(sample_text)
    
    # Print the result
    print("Input Text:", sample_text)
    print("NER Analysis:", result)
    