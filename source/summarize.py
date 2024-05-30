from transformers import BertTokenizer, EncoderDecoderModel

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("PaceKW/24PDInsight-Summarization")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

# Load the model
model = EncoderDecoderModel.from_pretrained("PaceKW/24PDInsight-Summarization")

def summarize_text(transcription):
    # Generate summary
    input_ids = tokenizer.encode(transcription, return_tensors='pt')
    summary_ids = model.generate(input_ids,
                                  min_length=20,
                                  max_length=80,
                                  num_beams=10,
                                  repetition_penalty=2.5,
                                  length_penalty=1.0,
                                  early_stopping=True,
                                  no_repeat_ngram_size=2,
                                  use_cache=True,
                                  do_sample=True,
                                  temperature=0.8,
                                  top_k=50,
                                  top_p=0.95)

    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# Test the function with your input
if __name__ == "__main__":
    sample_text = (
        "Kita harus bersyukur di tahun 2020 sampai 2030 nanti kita akan mendapatkan bonus demografi. "
        "Saat itulah sebagian besar penduduk kita ada pada usia produktif. Ini kesempatan kita untuk "
        "meningkatkan produktivitas nasional. Peluang untuk menuju Indonesia emas makin terbuka labor. "
        "Tapi Bapak-Ibu yang saya hormati, Teman-teman sesama anak muda, Ingat, kesempatan ini hanya "
        "datang sekali. Kesempatan ini tidak akan terulang lagi. Untuk itu kita harus kerja keras, kerja "
        "fokus, berani melakukan lompatan. Saya ucapkan terima kasih kepada Pak Prabowo, yang sudah "
        "memberi saya kesempatan untuk ikut andil dalam kontestasi ini. Saya sangat bangga sekali saya "
        "menjadi bagian dalam perjalanan menuju Indonesia Umas. Saya ucapkan terima kasih juga Prof. Mahfud, "
        "Guzmuhaymin. Saya sangat senang sekali bisa satu panggung dengan orang-orang hebat seperti ini. "
        "Senang sekali anak muda bisa bertukar pikiran dengan Ketua Umum Partai dan seorang Profesor. Sekali "
        "lagi terima kasih. Anak-anak muda harus saling mendukung, anak-anak muda harus saling bergandingan "
        "tangan. Saya yakin Indo Insyarmus bisa tercapai. Terima kasih. Wassalamu'alaikum warahmatullahi "
        "wabarakatuh. Selamat Natal dan Tahun Baru. Terima kasih telah menonton!"
    )
    summary = summarize_text(sample_text)
    print("Summary:", summary)
