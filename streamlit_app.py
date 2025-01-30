import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
import joblib

# Load attribut yg disimpan
tf_idf_vectorizer = joblib.load('vectorizer.pkl')
# Load model yg disimpan
svc = joblib.load('svc.pkl')


## Bersih bersih 
def cleansing(text):
    text = text.lower() # membuat semua teks menjadi huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', '', text) # menghapus semua kecuali huruf dan spasi
    text = re.sub(r'@\w+', '', text)  # Menghapus mention
    emoji_pattern = re.compile(
      "["
      u"\U0001F600-\U0001F64F"  # Emoticon
      u"\U0001F300-\U0001F5FF"  # Simbol & Objek
      u"\U0001F680-\U0001F6FF"  # Transportasi & Simbol
      u"\U0001F1E0-\U0001F1FF"  # Bendera
      u"\U00002702-\U000027B0"  # Simbol lain
      u"\U000024C2-\U0001F251"  # Simbol tambahan
      u"\U0001F910-\U0001F9FF"  # Emoji tambahan
      "]+",
      flags=re.UNICODE
  )
    text = emoji_pattern.sub(r'', text)
    return text


## Tokenize Stopwords
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
def tokenize_stopword(text):
    text = word_tokenize(text)
    text = [word for word in text if word not in stopwords]
    return text

## STEMMING
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def stemming(text):
    text = [stemmer.stem(word) for word in text]
    text = [lemmatizer.lemmatize(word) for word in text]
    return text


# Gabungin lagi
def gabungin_lagi(text):
    text = " ".join(text)
    return text

## TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf(text):
    tf_idf = tf_idf_vectorizer.transform([text])
    return tf_idf

# ==== BIKIN PIPELINE PROGRESS ====
def pipeline_progress(text):
    text = cleansing(text)
    st.write("After Cleansing : ")
    st.caption(text)

    text = tokenize_stopword(text)
    st.write("After Tokenize Stopword : ")
    st.caption(text)

    text = stemming(text)
    st.write("After Stemming : ")
    st.caption(text)

    text = gabungin_lagi(text)
    st.write("After Join : ")
    st.caption(text)

    return text



# ========== WEB APSS =============
st.title("News Classification with SVC Models")
st.write("This is a simple web app for news classification using SVC models")
st.caption("Create by : Nabil Yudis")

# Fungsi untuk mengambil input dari user
def predict_comment(text):
    # Process
    process_text = pipeline_progress(text)
    # TF-IDF
    tfidf_result = tfidf(process_text)
    st.text("Result of Matrix TF-IDF")
    st.write(tfidf_result)

    # Classification
    prediction_presentase = svc.predict(tfidf_result)
    st.text("Prediction Result")
    st.write(prediction_presentase)

    # # Classes
    # prediction_class = np.argmax(prediction_presentase, axis=1)
    # st.text("Prediction Classes")

    # Lupa lupa ini tidak perlu kan bukan Neural Network

    # Hasil Clasifikasi
    st.header("Classification Result")
    if prediction_presentase == 1:
        st.info("Class Index 1")
    elif prediction_presentase == 2:
        st.info("Class Index 2")
    elif prediction_presentase == 3:
        st.info("Class Index 3")
    else:
        st.info("Class Index 4")
    


tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔍 Classification", "ℹ️ About"])
with tab1:

    # Judul Halaman
    st.title("🚀 NLP Model Presentation")
    st.write("Selamat datang di presentasi model NLP! Di sini, saya akan membahas tahap-tahap pengembangan model saya dengan visualisasi yang menarik. 🎉")

    # **1. Class Index**
    st.header("📊 Class Distribution")
    st.write("Sebelum membangun model, kita perlu memahami distribusi kelas dalam dataset. Berikut adalah diagram lingkaran yang menunjukkan proporsi setiap kelas.")

    # Gambar Pie Chart Distribusi Kelas
    st.image("image/Pie Chart.png", caption="Distribusi Kelas", width=400)
 
    # **2. WordCloud Keseluruhan Data**
    st.header("☁ WordCloud Keseluruhan Data")
    st.write("Mari kita lihat kata-kata yang paling sering muncul dalam dataset! 🔍")

    # Gambar WordCloud Keseluruhan Data
    st.image("image/Wordcloud all.png", caption="WordCloud Keseluruhan Data", width=800)

    # **3. WordCloud Per Class**
    st.header("🎯 WordCloud Per Kelas")
    st.write("Berikut adalah visualisasi kata-kata utama dari masing-masing kelas.")
    st.image("image/Wordcloud index.png", caption="WordCloud Keseluruhan Data", width=800)


    # **4. Precision Score per Model**
    st.header("📊 Precision Score per Model")
    st.write("Berikut adalah nilai precision dari berbagai model yang saya coba untuk setiap kelas.")

    # Gambar Precision Score Table
    st.image("image/precision models.png", caption="Precision Score per Model",width=500)

    # **5. Barplot Precision Score**
    st.header("📊 Barplot Precision Score")
    st.write("Visualisasi ini membantu kita memahami perbandingan precision antar model.")

    # Gambar Precision Score Barplot
    st.image("image/barplot precision.png", caption="Barplot Precision Score", width=800)

    # **6. Score Train & Test**
    st.header("🏋️‍♂️ Score Train & Test")
    st.write("Bagaimana performa model saat dilatih dan diuji? Ini datanya! 📈")

    # Gambar Score Train vs Test Table
    st.image("image/score models.png", caption="Score Train & Test", width=350)

    # **7. Barplot Score Training & Testing**
    st.header("📊 Perbandingan Score Train vs Test")

    # Gambar Score Train vs Test Barplot
    st.image("image/barplot score models.png", caption="Barplot Score Train vs Test",width=650)

    # **8. Accuracy Validation Score**
    st.header("🧪 Validasi Akurasi Model")
    st.write("Bagaimana performa model pada dataset baru yang belum pernah dilihat? Ini hasil validasinya! 🎯")

    # Gambar Accuracy Validation Table
    st.image("image/accuracy val.png", caption="Accuracy Validation Table", width=280)

    # **9. Barplot Accuracy Validation**
    st.header("📊 Visualisasi Akurasi Model pada Dataset Validasi")

    # Gambar Accuracy Validation Barplot
    st.image("image/barplot accuracy val.png", caption="Barplot Accuracy Validation",)

    # Penutup
    st.write("🎉 Sekian presentasi saya mengenai proses pengembangan model ini. Semoga bermanfaat! Jangan ragu untuk mencoba fitur prediksi di tab **Classification** ya! 🚀")

    st.info("Silakan menuju tab 'Classification' untuk mencoba fitur ini.")

with tab2:
# Bikin Button
    input_text_user = st.text_area("Input your text here : ")
    button = st.button("Submit")
    if button:
        if input_text_user:
            st.text("Your Text : ")
            st.caption(input_text_user)
            result_final = predict_comment(input_text_user)
            st.write(result_final)
        else:
            st.write("Please input your text first")

with tab3:
    st.write("About")
