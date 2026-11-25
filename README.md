# 📄 RAG Soru-Cevap Sistemi

Yüklenen bir PDF belgesi hakkında Türkçe soru sorup, cevabı belgenin kendisinden kaynak göstererek alan bir **RAG (Retrieval-Augmented Generation)** uygulaması.

## ✨ Özellikler

- PDF yükleme ve sayfa sayfa metin çıkarma (PyMuPDF)
- Bozuk Türkçe karakterlerin düzeltilmesi
- Cümle sınırına duyarlı, örtüşmeli parçalama (chunking)
- Çok dilli embedding modeli: `intfloat/multilingual-e5-small`
- Kosinüs benzerliği ile en ilgili parçaların bulunması (top-k)
- Google Gemini ile yalnızca belgeye dayalı cevap üretimi
- Cevapta kaynak numaraları ([1], [2]…) ve sayfa bilgisi
- Ayarlanabilir parametreler: parça boyutu, örtüşme, top-k, benzerlik eşiği

## ⚙️ Nasıl Çalışır?

1. PDF okunur ve metin parçalara bölünür.
2. Her parça embedding vektörüne dönüştürülür.
3. Soru da vektöre çevrilir ve en benzer `k` parça seçilir.
4. Seçilen parçalar ve soru Gemini modeline gönderilir.
5. Model, sadece bu parçalardaki bilgiye dayanarak cevap verir. Cevap yoksa "bulamadım" der.

## 🚀 Kurulum

```bash
git clone https://github.com/KULLANICI_ADI/rag-soru-cevap.git
cd rag-soru-cevap
pip install -r requirements.txt
```

[Google AI Studio](https://aistudio.google.com/)'dan bir API anahtarı alın. `.streamlit/secrets.toml.example` dosyasını `.streamlit/secrets.toml` olarak kopyalayıp anahtarınızı yazın:

```toml
GEMINI_API_KEY = "sizin-anahtariniz"
```

Uygulamayı çalıştırın:

```bash
streamlit run app.py
```

## 🛠️ Kullanılan Teknolojiler

- [Streamlit](https://streamlit.io/): Arayüz
- [PyMuPDF](https://pymupdf.readthedocs.io/): PDF okuma
- [Sentence Transformers](https://www.sbert.net/): Embedding
- [Google Gemini API](https://ai.google.dev/): Cevap üretimi
- NumPy: Vektör araması
