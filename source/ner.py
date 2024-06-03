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

def split_text_into_segments(text, tokenizer, max_length=512):
    tokens = tokenizer(text, return_tensors="pt", padding=False, truncation=False)["input_ids"][0]
    num_tokens = len(tokens)
    segments = []
    for i in range(0, num_tokens, max_length):
        segment = tokens[i:i + max_length]
        segments.append(segment)
    return segments

def ner_predict_for_segments(segments, model, tokenizer, label_translation):
    predictions = []
    for segment in segments:
        inputs = {"input_ids": torch.unsqueeze(segment, 0)}
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_labels = torch.argmax(probabilities, dim=-1)

        words = tokenizer.convert_ids_to_tokens(segment)
        labels = [model.config.id2label[label_id.item()] for label_id in predicted_labels]

        translated_labels = [label_translation[label] for label in labels]
        result = [(word, label) for word, label in zip(words, translated_labels) if word not in tokenizer.all_special_tokens]
        predictions.extend(result)
    return predictions

def ner_format_result(result):
    formatted_text = ""
    for word, label in result:
        if label != 'O':
            formatted_text += f"{word} [{label}] "
        else:
            formatted_text += f"{word} "
    return formatted_text.strip()

def analyze_ner(text):
    segments = split_text_into_segments(text, ner_tokenizer)
    ner_result = ner_predict_for_segments(segments, ner_model, ner_tokenizer, label_translation)
    formatted_ner_result = ner_format_result(ner_result)
    return formatted_ner_result



if __name__ == "__main__":
    # Sample text for testing
    sample_text = "Kita harus bersyukur di tahun 2020 sampai 2030 nanti kita akan mendapatkan bonus demografi. Saat itulah sebagian besar penduduk kita ada pada usia produktif. Ini kesempatan kita untuk meningkatkan produktivitas nasional. Peluang untuk menuju Indonesia emas makin terbuka lebar. Tapi Bapak-Ibu yang saya hormati, teman-teman sesama anak muda, ingat, kesempatan ini hanya datang sekali. Kesempatan ini tidak akan terulang lagi. Untuk itu kita harus kerja keras, kerja fokus, berani melakukan lompatan. Saya ucapkan terima kasih kepada Pak Prabowo, yang sudah memberi saya kesempatan untuk ikut andil dalam kontestasi ini. Saya sangat bangga sekali saya menjadi bagian dalam perjalanan menuju Indonesia emas. Saya ucapkan terima kasih juga Prof. Mahfud, Guzmuhaymin. Saya sangat senang sekali bisa satu panggung dengan orang-orang hebat seperti ini. Senang sekali anak muda bisa bertukar pikiran dengan Ketua Umum Partai dan seorang Profesor. Sekali lagi terima kasih. Anak-anak muda harus saling mendukung, anak-anak muda harus saling bergandengan tangan. Saya yakin Indo Insyarmus bisa tercapai. Terima kasih. Wassalamu'alaikum warahmatullahi wabarakatuh. Selamat Natal dan Tahun Baru. Terima kasih telah menonton!"

    # Analyze the sample text
    result = analyze_ner(sample_text)

    # Print the result
    print("Input Text:", sample_text)
    print("NER Analysis:", result)
