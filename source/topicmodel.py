import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim import corpora
from gensim.models import LdaModel
import nltk
import json

# Download NLTK punkt tokenizer resource
nltk.download('punkt')

def topic_modeling(transcription):
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        text = text.strip()  # Remove leading and trailing whitespaces
        text = re.sub('\s+', ' ', text)  # Remove multiple whitespaces
        text = re.sub(r"\b[a-zA-Z]\b", "", text)  # Remove single characters

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in list_stopwords]  # Remove stopwords
        tokens = [stemmer.stem(word) for word in tokens]  # Stemming

        return tokens

    # Download NLTK stopwords resource
    nltk.download('stopwords')
    list_stopwords = stopwords.words('indonesian')
    additional_stopwords = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                            'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                            'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                            '&amp', 'yah', 'bisnis', 'pandemi', 'indonesia', 'saya', 'sekali', 'ini', 'harus',
                            'sempat', 'terima', 'kasih', 'untuk']
    list_stopwords.extend(additional_stopwords)
    list_stopwords = set(list_stopwords)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    transcription_tokens = preprocess_text(transcription)
    dictionary = corpora.Dictionary([transcription_tokens])
    doc_term_matrix = [dictionary.doc2bow(transcription_tokens)]

    total_topics = 2
    lda_model = LdaModel(doc_term_matrix, num_topics=total_topics, id2word=dictionary, passes=50)
    topics = lda_model.show_topics(num_topics=total_topics, num_words=5)

    # Prepare data for JSON
    json_data = []
    for topic_id, topic in topics:
        keywords = [word.strip().split("*")[1].strip('"') for word in topic.split("+")]
        importance = [float(word.strip().split("*")[0]) for word in topic.split("+")]
        word_count = [dictionary.token2id[word] for word in keywords]  # Get word count frequency
        for i in range(len(keywords)):
            json_data.append({
                "word": keywords[i],
                "topic_id": topic_id,
                "importance": importance[i],
                "word_count": word_count[i]
            })

    return json_data

if __name__ == "__main__":
    # Sample transcription
    transcription = """
    Kita harus bersyukur di tahun 2020 sampai 2030 nanti kita akan mendapatkan bonus demografi. 
    Saat itulah sebagian besar penduduk kita ada pada usia produktif. Ini kesempatan kita untuk 
    meningkatkan produktivitas nasional. Peluang untuk menuju Indonesia emas makin terbuka labor. 
    Tapi Bapak-Ibu yang saya hormati, Teman-teman sesama anak muda, Ingat, kesempatan ini hanya 
    datang sekali. Kesempatan ini tidak akan terulang lagi. Untuk itu kita harus kerja keras, kerja 
    fokus, berani melakukan lompatan. Saya ucapkan terima kasih kepada Pak Prabowo, yang sudah 
    memberi saya kesempatan untuk ikut andil dalam kontestasi ini. Saya sangat bangga sekali saya 
    menjadi bagian dalam perjalanan menuju Indonesia Umas. Saya ucapkan terima kasih juga Prof. Mahfud, 
    Guzmuhaymin. Saya sangat senang sekali bisa satu panggung dengan orang-orang hebat seperti ini. 
    Senang sekali anak muda bisa bertukar pikiran dengan Ketua Umum Partai dan seorang Profesor. Sekali 
    lagi terima kasih. Anak-anak muda harus saling mendukung, anak-anak muda harus saling bergandingan 
    tangan. Saya yakin Indo Insyarmus bisa tercapai. Terima kasih. Wassalamu'alaikum warahmatullahi 
    wabarakatuh. Selamat Natal dan Tahun Baru. Terima kasih telah menonton!
    """

    # Test the function
    topics = topic_modeling(transcription)
    print("Topics:", topics)
