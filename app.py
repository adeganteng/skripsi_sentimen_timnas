import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import io
from fpdf import FPDF

# ==========================================
# 1. SETUP PAGE & KONFIGURASI
# ==========================================
st.set_page_config(page_title="Sentimen Timnas", page_icon="⚽", layout="wide")

# ==========================================
# 2. FUNGSI CACHING & PREPROCESSING
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan nama folder ini sama persis dengan folder hasil ekstrak ZIP lu
    model_path = "./model_indobert_timnas" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Load model ke dalam memory web
tokenizer, model = load_model()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 3. SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    st.title("⚽ Navigasi Utama")
    menu = st.radio(
        "Pilih Mode Analisis:",
        ("1. Analisis Komentar Tunggal", "2. Analisis Dataset", "3. Evaluasi Model")
    )
    st.markdown("---")
    st.write("© 2026 - Ade Wicaksono")

# ==========================================
# 4. LOGIKA HALAMAN MENU
# ==========================================

# ----------------------------------------------------
# MENU 1: ANALISIS KOMENTAR TUNGGAL
# ----------------------------------------------------
if menu == "1. Analisis Komentar Tunggal":
    st.title("🔍 Analisis Komentar Tunggal")
    st.write("Masukkan 1 komentar terkait Timnas Indonesia untuk dianalisis oleh model IndoBERT.")

    # Input Box dari user
    user_input = st.text_area("Masukkan Komentar:", placeholder="Contoh: Timnas mainnya bagus banget hari ini, bangga!")

    # Tombol Eksekusi
    if st.button("Analisis Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("Komentarnya jangan kosong dong, bro!")
        else:
            with st.spinner("Model sedang berpikir..."):
                # A. Preprocessing
                clean_text = preprocess_text(user_input)

                # B. Tokenization
                inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

                # C. Prediksi dengan Model
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1).squeeze()

                # Ambil probabilitas untuk ketiga kelas (urutan: 0=Positif, 1=Netral, 2=Negatif)
                prob_positif = probs[0].item() * 100
                prob_netral = probs[1].item() * 100
                prob_negatif = probs[2].item() * 100

                # Dapatkan class ID pemenang
                predicted_class_id = torch.argmax(probs).item()
                label_map = {0: "Positif", 1: "Netral", 2: "Negatif"}
                sentiment = label_map[predicted_class_id]
                
                # Confidence score pemenang (dalam persen)
                confidence_score = max(prob_positif, prob_netral, prob_negatif)

                # D. Logika Penjelasan Dinamis (Summary)
                alasan_kalimat = ""
                alasan_angka = ""

                # 1. Alasan kenapa masuk kelas tersebut
                if sentiment == "Positif":
                    alasan_kalimat = "Berdasarkan pola bahasa yang dipelajari selama pelatihan, model mendeteksi adanya kosakata atau frasa yang menunjukkan dukungan, pujian, atau rasa bangga terhadap performa Timnas Indonesia."
                elif sentiment == "Negatif":
                    alasan_kalimat = "Model mengidentifikasi adanya pola kalimat yang mengandung kritik, kekecewaan, cacian, atau nada pesimis terhadap pemain, pelatih, maupun permainan Timnas."
                else:
                    alasan_kalimat = "Model tidak menemukan kecenderungan emosi yang kuat. Kalimat ini diklasifikasikan sebagai netral karena sifatnya yang informatif, berupa pertanyaan biasa, atau sekadar komentar spam yang tidak memihak."

                # 2. Alasan kenapa nilainya segitu (Berdasarkan Confidence Score)
                if confidence_score >= 90:
                    alasan_angka = f"Model sangat yakin (confidence: {confidence_score:.1f}%) dengan prediksi ini karena emosi dalam kalimat sangat eksplisit dan mudah dikenali."
                elif confidence_score >= 70:
                    alasan_angka = f"Model cukup yakin (confidence: {confidence_score:.1f}%) dengan prediksi ini. Meskipun begitu, masih ada sedikit ambiguitas atau gaya bahasa gaul yang membuat probabilitasnya tidak mencapai maksimal."
                else:
                    # Kalau confidence rendah, berarti model agak bingung antara 2 label
                    label_terbesar_kedua = ""
                    if sentiment == "Positif":
                        label_terbesar_kedua = "Netral" if prob_netral > prob_negatif else "Negatif"
                    elif sentiment == "Negatif":
                        label_terbesar_kedua = "Netral" if prob_netral > prob_positif else "Positif"
                    else:
                        label_terbesar_kedua = "Positif" if prob_positif > prob_negatif else "Negatif"
                        
                    alasan_angka = f"Model kurang yakin (confidence hanya {confidence_score:.1f}%). Hal ini terjadi karena gaya bahasa yang digunakan mengandung unsur sarkasme, singkatan yang rumit, atau emosi yang campur aduk, sehingga model juga melihat adanya potensi sentimen {label_terbesar_kedua} pada komentar ini."

                # Gabungkan penjelasan
                penjelasan_lengkap = f"{alasan_kalimat} {alasan_angka}"

                # E. Tampilkan Hasil (UI Output)
                st.markdown("---")
                st.subheader("📊 Hasil Analisis")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Prediksi Sentimen", value=sentiment)
                with col2:
                    # Menampilkan confidence score dalam bentuk Persen
                    st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence_score:.1f}%")

                st.info(f"**💡 Mengapa hasilnya demikian?**\n\n{penjelasan_lengkap}")
                
                # Opsional: Tampilkan detail probabilitas ketiga kelas biar makin ilmiah
                with st.expander("Lihat Detail Probabilitas Semua Kelas"):
                    st.write(f"- 🟢 Positif: {prob_positif:.2f}%")
                    st.write(f"- ⚪ Netral: {prob_netral:.2f}%")
                    st.write(f"- 🔴 Negatif: {prob_negatif:.2f}%")
                    
# ----------------------------------------------------
# MENU 2: ANALISIS DATASET (TANPA LABEL)
# ----------------------------------------------------
# ----------------------------------------------------
# MENU 2: ANALISIS DATASET (TANPA LABEL)
# ----------------------------------------------------
elif menu == "2. Analisis Dataset":
    st.title("📊 Analisis Dataset Komentar")
    st.write("Upload file CSV/Excel tanpa label. Sistem akan mengklasifikasikan, mengukur probabilitas, dan membuatkan laporan otomatis.")

    # 1. Fitur Upload File
    uploaded_file = st.file_uploader("📂 Upload file dataset (Pastikan ada kolom 'komentar')", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        if 'komentar' not in df.columns:
            st.error("❌ Gagal! File harus memiliki kolom dengan nama 'komentar' (huruf kecil semua).")
        else:
            st.success(f"✅ File berhasil diunggah! Total data: {len(df)} baris.")
            
            if st.button("🚀 Mulai Analisis Massal & Buat Laporan", type="primary"):
                
                progress_text = "AI sedang membaca, memprediksi, dan mengkalkulasi probabilitas. Mohon tunggu..."
                my_bar = st.progress(0, text=progress_text)
                
                sentimens = []
                label_ints = [] # Menyimpan angka 0, 1, 2
                prob_pos_list = []
                prob_net_list = []
                prob_neg_list = []
                
                # Looping Prediksi dan Ekstraksi Probabilitas
                for i, row in df.iterrows():
                    teks = str(row['komentar'])
                    teks_bersih = preprocess_text(teks)
                    
                    inputs = tokenizer(teks_bersih, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = F.softmax(outputs.logits, dim=1).squeeze()
                    
                    p_pos = probs[0].item() * 100
                    p_net = probs[1].item() * 100
                    p_neg = probs[2].item() * 100
                    
                    pred_id = torch.argmax(probs).item()
                    label_map = {0: "Positif", 1: "Netral", 2: "Negatif"}
                    
                    # Simpan data ke list
                    sentimens.append(label_map[pred_id])
                    label_ints.append(pred_id) # Menyimpan ID murni (0, 1, 2)
                    prob_pos_list.append(f"{p_pos:.1f}%")
                    prob_net_list.append(f"{p_net:.1f}%")
                    prob_neg_list.append(f"{p_neg:.1f}%")
                    
                    my_bar.progress((i + 1) / len(df), text=f"Memproses baris {i+1} dari {len(df)}...")
                
                my_bar.empty()
                
                # Masukkan list ke dalam DataFrame
                df['Prediksi_Sentimen'] = sentimens
                df['label'] = label_ints # Kolom baru berisi 0, 1, atau 2
                df['Prob_Positif'] = prob_pos_list
                df['Prob_Netral'] = prob_net_list
                df['Prob_Negatif'] = prob_neg_list
                
                st.markdown("---")
                st.header("🎯 Hasil Analisis & Visualisasi")
                
                # ==========================================
                # VISUALISASI 1: PIE CHART & BAR CHART
                # ==========================================
                st.subheader("📈 Distribusi Sentimen Opini Publik")
                fig_col1, fig_col2 = st.columns(2)
                warna_sentimen = {'Positif': '#2ecc71', 'Netral': '#95a5a6', 'Negatif': '#e74c3c'}
                
                with fig_col1:
                    fig_pie, ax_pie = plt.subplots()
                    df['Prediksi_Sentimen'].value_counts().plot.pie(
                        autopct='%1.1f%%', ax=ax_pie, 
                        colors=[warna_sentimen.get(x, '#333333') for x in df['Prediksi_Sentimen'].value_counts().index],
                        startangle=90, shadow=True
                    )
                    ax_pie.set_ylabel('') 
                    st.pyplot(fig_pie)
                    fig_pie.savefig("temp_pie.png", bbox_inches='tight')
                    
                with fig_col2:
                    fig_bar, ax_bar = plt.subplots()
                    sns.countplot(data=df, x='Prediksi_Sentimen', hue='Prediksi_Sentimen', palette=warna_sentimen, ax=ax_bar, order=["Positif", "Netral", "Negatif"], legend=False)
                    ax_bar.set_ylabel('Jumlah Komentar')
                    ax_bar.set_xlabel('Kategori Sentimen')
                    st.pyplot(fig_bar)
                    fig_bar.savefig("temp_bar.png", bbox_inches='tight')

                # ==========================================
                # VISUALISASI 2: WORDCLOUD PER SENTIMEN
                # ==========================================
                st.markdown("---")
                st.subheader("☁️ Wordcloud (Kata Populer per Sentimen)")
                tab_pos, tab_net, tab_neg = st.tabs(["🟢 Positif", "⚪ Netral", "🔴 Negatif"])
                
                def buat_wordcloud_dan_save(teks_series, background_color, colormap, filename):
                    if teks_series.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.text(0.5, 0.5, 'Tidak Ada Data', horizontalalignment='center', verticalalignment='center', fontsize=20)
                        ax.axis('off')
                        fig.savefig(filename, bbox_inches='tight')
                        return fig
                    
                    teks_gabungan = " ".join(preprocess_text(str(teks)) for teks in teks_series)
                    wc = WordCloud(width=800, height=400, background_color=background_color, colormap=colormap, max_words=100).generate(teks_gabungan)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    fig.savefig(filename, bbox_inches='tight')
                    return fig

                with tab_pos:
                    fig_pos = buat_wordcloud_dan_save(df[df['Prediksi_Sentimen'] == 'Positif']['komentar'], 'white', 'Greens', "temp_wc_pos.png")
                    st.pyplot(fig_pos)
                with tab_net:
                    fig_net = buat_wordcloud_dan_save(df[df['Prediksi_Sentimen'] == 'Netral']['komentar'], 'white', 'Greys', "temp_wc_net.png")
                    st.pyplot(fig_net)
                with tab_neg:
                    fig_neg = buat_wordcloud_dan_save(df[df['Prediksi_Sentimen'] == 'Negatif']['komentar'], 'white', 'Reds', "temp_wc_neg.png")
                    st.pyplot(fig_neg)

                # ==========================================
                # TABEL DATA & KESIMPULAN
                # ==========================================
                st.markdown("---")
                st.subheader("📑 Preview Dataset Lengkap (Dengan Label & Probabilitas)")
                kolom_tampil = ['komentar', 'Prediksi_Sentimen', 'Prob_Positif', 'Prob_Netral', 'Prob_Negatif', 'label']
                st.dataframe(df[kolom_tampil].head(10))

                st.markdown("---")
                st.subheader("📝 Kesimpulan Analisis Otomatis")
                mayoritas = df['Prediksi_Sentimen'].value_counts().idxmax()
                persentase_mayoritas = (df['Prediksi_Sentimen'].value_counts().max() / len(df)) * 100
                
                opini = "memberikan dukungan, apresiasi, dan respon positif" if mayoritas == "Positif" else "mengungkapkan kekecewaan, kritik, atau respon negatif" if mayoritas == "Negatif" else "memberikan respon yang netral, informatif, atau campuran"
                teks_kesimpulan = f"Berdasarkan analisis terhadap {len(df)} komentar, sentimen publik mayoritas menunjukkan sentimen {mayoritas} ({persentase_mayoritas:.1f}%). Hal ini menunjukkan bahwa sebagian besar pengguna Instagram {opini} terhadap dinamika Timnas Indonesia."
                st.success(teks_kesimpulan)

                # ==========================================
                # GENERATE LAPORAN PDF (ALL-IN-ONE)
                # ==========================================
                pdf = FPDF()
                pdf.add_page()
                
                # Header
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, txt="Laporan Analisis Dataset Opini Publik", ln=True, align='C')
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, txt="Topik: Sentimen Komentar Instagram @timnasindonesia", ln=True, align='C')
                pdf.ln(5)
                
                # 1. Metrik Singkat
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, txt=f"Total Data Dianalisis: {len(df)} Komentar", ln=True)
                pdf.set_font("Arial", size=11)
                pdf.cell(0, 6, txt=f"- Positif : {len(df[df['Prediksi_Sentimen'] == 'Positif'])} data", ln=True)
                pdf.cell(0, 6, txt=f"- Netral  : {len(df[df['Prediksi_Sentimen'] == 'Netral'])} data", ln=True)
                pdf.cell(0, 6, txt=f"- Negatif : {len(df[df['Prediksi_Sentimen'] == 'Negatif'])} data", ln=True)
                pdf.ln(5)
                
                # 2. Grafik Bar & Pie (Diatur bersebelahan)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, txt="1. Distribusi Sentimen (Pie Chart & Bar Chart):", ln=True)
                pdf.image("temp_pie.png", x=10, w=90)
                pdf.image("temp_bar.png", x=110, y=pdf.get_y()-65, w=90)
                pdf.ln(5)
                
                # Halaman Baru untuk Wordcloud agar tidak bertumpuk
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, txt="2. Visualisasi Wordcloud (Kata Paling Sering Muncul):", ln=True)
                
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(0, 6, txt="A. Sentimen Positif:", ln=True)
                pdf.image("temp_wc_pos.png", w=140)
                pdf.ln(2)
                
                pdf.cell(0, 6, txt="B. Sentimen Negatif:", ln=True)
                pdf.image("temp_wc_neg.png", w=140)
                pdf.ln(2)
                
                pdf.cell(0, 6, txt="C. Sentimen Netral:", ln=True)
                pdf.image("temp_wc_net.png", w=140)
                
                # Halaman Baru untuk Tabel & Summary
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, txt="3. Preview Data Klasifikasi & Probabilitas (Top 5):", ln=True)
                pdf.set_font("Arial", size=9)
                pdf.cell(0, 6, txt="*Catatan: Ini hanya pratinjau. Unduh file Excel/CSV untuk melihat seluruh data.", ln=True)
                pdf.ln(3)
                
                # Header Tabel PDF (Disesuaikan dengan format baru)
                pdf.set_fill_color(200, 220, 255)
                pdf.set_font("Arial", 'B', 8)
                pdf.cell(85, 8, "Komentar (Dipotong)", border=1, fill=True)
                pdf.cell(25, 8, "Prediksi", border=1, fill=True, align='C')
                pdf.cell(15, 8, "Label", border=1, fill=True, align='C')
                pdf.cell(25, 8, "Prob Pos", border=1, fill=True, align='C')
                pdf.cell(25, 8, "Prob Neg", border=1, fill=True, align='C')
                pdf.ln()
                
                # Fungsi pembersih karakter untuk FPDF agar tidak error
                def sanitize_fpdf(text):
                    return str(text).encode('latin-1', 'replace').decode('latin-1')

                # Isi Tabel (5 Baris Pertama)
                pdf.set_font("Arial", size=8)
                for index, row in df.head(5).iterrows():
                    teks_potong = str(row['komentar'])[:45] + "..." if len(str(row['komentar'])) > 45 else str(row['komentar'])
                    
                    # Sanitize teks sebelum masuk PDF
                    teks_aman = sanitize_fpdf(teks_potong)
                    
                    pdf.cell(85, 8, teks_aman, border=1)
                    pdf.cell(25, 8, str(row['Prediksi_Sentimen']), border=1, align='C')
                    pdf.cell(15, 8, str(row['label']), border=1, align='C')
                    pdf.cell(25, 8, str(row['Prob_Positif']), border=1, align='C')
                    pdf.cell(25, 8, str(row['Prob_Negatif']), border=1, align='C')
                    pdf.ln()
                
                pdf.ln(10)
                
                # Kesimpulan Akhir
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, txt="4. Kesimpulan:", ln=True)
                pdf.set_font("Arial", size=11)
                
                # Sanitize kesimpulan just in case
                teks_kesimpulan_aman = sanitize_fpdf(teks_kesimpulan)
                pdf.multi_cell(0, 6, txt=teks_kesimpulan_aman)

                pdf_bytes = pdf.output(dest='S').encode('latin-1')

                # ==========================================
                # DOWNLOAD SECTION (3 TOMBOL BERDERET)
                # ==========================================
                st.markdown("---")
                st.subheader("💾 Unduh Hasil Lengkap")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1: # Tombol PDF
                    st.download_button("📥 Laporan PDF Lengkap", data=pdf_bytes, file_name='Laporan_Analisis_Timnas.pdf', mime='application/pdf', type='primary', use_container_width=True)
                with col_dl2: # Tombol CSV
                    csv_hasil = df[kolom_tampil].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Data CSV (.csv)", data=csv_hasil, file_name='dataset_timnas_berlabel.csv', mime='text/csv', use_container_width=True)
                with col_dl3: # Tombol Excel
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df[kolom_tampil].to_excel(writer, index=False, sheet_name='Hasil Sentimen')
                    st.download_button("📥 Data Excel (.xlsx)", data=excel_buffer.getvalue(), file_name='dataset_timnas_berlabel.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
                    
# ----------------------------------------------------
# MENU 3: EVALUASI PERFORMA MODEL
# ----------------------------------------------------
elif menu == "3. Evaluasi Model":
    st.title("📈 Evaluasi Performa Model IndoBERT")
    st.write("Upload dataset pengujian (Testing) yang **sudah memiliki label asli** untuk mengukur tingkat akurasi model.")

    st.info("⚠️ Pastikan file memiliki kolom 'komentar' dan kolom 'label' (berisi angka 0=Positif, 1=Netral, 2=Negatif).")
    
    # 1. Fitur Upload Dataset Pengujian
    uploaded_test_file = st.file_uploader("📂 Upload dataset testing", type=["csv", "xlsx", "xls"], key="eval_upload")

    if uploaded_test_file is not None:
        if uploaded_test_file.name.endswith('.csv'):
            df_test = pd.read_csv(uploaded_test_file)
        else:
            df_test = pd.read_excel(uploaded_test_file)
            
        if 'komentar' not in df_test.columns or 'label' not in df_test.columns:
            st.error("❌ File harus memiliki kolom 'komentar' dan 'label'.")
        else:
            st.success(f"✅ Dataset berhasil dimuat! Total: {len(df_test)} baris data uji.")
            
            if st.button("🚀 Mulai Evaluasi Model", type="primary"):
                with st.spinner("Model sedang memprediksi dan membandingkan hasil..."):
                    
                    y_true = []
                    y_pred = []
                    
                    for i, row in df_test.iterrows():
                        y_true.append(int(row['label']))
                        teks_bersih = preprocess_text(str(row['komentar']))
                        inputs = tokenizer(teks_bersih, return_tensors="pt", truncation=True, padding=True, max_length=128)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            pred_id = torch.argmax(F.softmax(outputs.logits, dim=1)).item()
                        y_pred.append(pred_id)

                    # Kalkulasi Metrik & Matriks
                    acc = accuracy_score(y_true, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                    total_data = len(df_test)
                    
                    # ==========================================
                    # 1. VISUALISASI CONFUSION MATRIX 
                    # ==========================================
                    st.markdown("---")
                    st.subheader("🧩 1. Visualisasi Confusion Matrix")
                    st.write("Detail sebaran tebakan model. Angka di garis diagonal adalah tebakan yang benar.")
                    
                    label_names = ['Positif', 'Netral', 'Negatif']
                    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax_cm)
                    ax_cm.set_ylabel('Label Aktual (Asli)')
                    ax_cm.set_xlabel('Label Prediksi (Model)')
                    st.pyplot(fig_cm)
                    fig_cm.savefig("temp_cm.png")

                    # ==========================================
                    # 2. RINCIAN ANGKA MATRIKS PER KELAS (BARU!)
                    # ==========================================
                    # Ekstrak angka TP, FP, FN per kelas dari matriks
                    # Format: cm[baris][kolom]
                    
                    # Kelas POSITIF (Index 0)
                    tp_pos = cm[0][0]
                    fn_pos = cm[0][1] + cm[0][2] # Aktual positif tapi ditebak netral/negatif
                    fp_pos = cm[1][0] + cm[2][0] # Aktual netral/negatif tapi ditebak positif
                    
                    # Kelas NETRAL (Index 1)
                    tp_net = cm[1][1]
                    fn_net = cm[1][0] + cm[1][2]
                    fp_net = cm[0][1] + cm[2][1]
                    
                    # Kelas NEGATIF (Index 2)
                    tp_neg = cm[2][2]
                    fn_neg = cm[2][0] + cm[2][1]
                    fp_neg = cm[0][2] + cm[1][2]

                    total_benar = tp_pos + tp_net + tp_neg

                    st.markdown("##### 📌 Rincian Nilai Evaluasi Tiap Sentimen:")
                    col_p, col_nt, col_ng = st.columns(3)
                    
                    with col_p:
                        st.info(f"**🟢 Sentimen Positif**\n\n- **True Positif (TP):** {tp_pos} *(Tebakan Tepat)*\n- **False Positif (FP):** {fp_pos} *(Salah nebak sbg Positif)*\n- **False Negatif (FN):** {fn_pos} *(Gagal nembak Positif)*")
                    with col_nt:
                        st.warning(f"**⚪ Sentimen Netral**\n\n- **True Netral (TP):** {tp_net} *(Tebakan Tepat)*\n- **False Netral (FP):** {fp_net} *(Salah nebak sbg Netral)*\n- **False Negatif (FN):** {fn_net} *(Gagal nembak Netral)*")
                    with col_ng:
                        st.error(f"**🔴 Sentimen Negatif**\n\n- **True Negatif (TP):** {tp_neg} *(Tebakan Tepat)*\n- **False Negatif (FP):** {fp_neg} *(Salah nebak sbg Negatif)*\n- **False Negatif (FN):** {fn_neg} *(Gagal nembak Negatif)*")

                    # ==========================================
                    # 3. HASIL EVALUASI METRIK 
                    # ==========================================
                    st.markdown("---")
                    st.subheader("📊 2. Hasil Evaluasi Metrik (Macro-Average)")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Akurasi (Accuracy)", f"{acc*100:.1f}%")
                    col2.metric("Presisi (Precision)", f"{precision*100:.1f}%")
                    col3.metric("Recall", f"{recall*100:.1f}%")
                    col4.metric("F1-Score", f"{f1*100:.1f}%")

                    # ==========================================
                    # 4. PEMBEDAHAN SEMUA RUMUS 
                    # ==========================================
                    st.markdown("---")
                    st.subheader("🧮 3. Pembedahan Rumus Evaluasi")
                    st.write("💡 Karena terdapat 3 label (Multiclass), sistem menggunakan pendekatan **Macro-Average**.")
                    
                    col_rumus1, col_rumus2 = st.columns(2)
                    
                    with col_rumus1:
                        st.write("**1. Accuracy (Akurasi)**")
                        st.latex(r"Accuracy = \frac{\text{Total Prediksi Benar}}{\text{Total Data}}")
                        st.latex(rf"Accuracy = \frac{{{total_benar}}}{{{total_data}}} = {acc*100:.1f}\%")
                        
                        st.write("**2. Precision (Presisi)**")
                        st.latex(r"Precision = \frac{TP}{TP + FP}")
                        st.latex(rf"Precision_{{Macro}} = {precision*100:.1f}\%")

                    with col_rumus2:
                        st.write("**3. Recall (Sensitivitas)**")
                        st.latex(r"Recall = \frac{TP}{TP + FN}")
                        st.latex(rf"Recall_{{Macro}} = {recall*100:.1f}\%")
                        
                        st.write("**4. F1-Score**")
                        st.latex(r"F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}")
                        st.latex(rf"F1_{{Macro}} = {f1*100:.1f}\%")

                    # Kesimpulan
                    kategori_performa = "sangat baik dan sangat akurat" if acc >= 0.85 else "baik dan cukup akurat" if acc >= 0.75 else "masih memerlukan perbaikan"
                    teks_kesimpulan = f"Berdasarkan pengujian terhadap {total_data} data uji, model IndoBERT menunjukkan performa yang {kategori_performa} dengan nilai akurasi mencapai {acc*100:.1f}%. Rincian matriks menunjukkan kemampuan model dalam membedakan ketiga kelas sentimen secara objektif."
                    st.success(f"**Kesimpulan:**\n{teks_kesimpulan}")

                    # ==========================================
                    # 5. FITUR EXPORT LAPORAN PDF
                    # ==========================================
                    st.markdown("---")
                    st.subheader("📄 4. Cetak Laporan Evaluasi Full")
                    
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Judul
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, txt="Laporan Evaluasi Performa Model IndoBERT", ln=True, align='C')
                    pdf.set_font("Arial", size=12)
                    pdf.cell(0, 10, txt="Analisis Sentimen Komentar Instagram @timnasindonesia", ln=True, align='C')
                    pdf.ln(5)
                    
                    # 1. Gambar CM
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt="1. Visualisasi Confusion Matrix:", ln=True)
                    pdf.image("temp_cm.png", x=30, w=150) 
                    pdf.ln(2)
                    
                    # 2. Rincian Matriks (Dimasukkan ke PDF)
                    pdf.set_font("Arial", 'B', 11)
                    pdf.cell(0, 8, txt="   Rincian Nilai per Kelas:", ln=True)
                    pdf.set_font("Arial", size=11)
                    # Positif
                    pdf.cell(0, 6, txt=f"   - POSITIF : Benar (TP)={tp_pos} | Salah sbg Positif (FP)={fp_pos} | Gagal tebak Positif (FN)={fn_pos}", ln=True)
                    # Netral
                    pdf.cell(0, 6, txt=f"   - NETRAL  : Benar (TP)={tp_net} | Salah sbg Netral (FP)={fp_net} | Gagal tebak Netral (FN)={fn_net}", ln=True)
                    # Negatif
                    pdf.cell(0, 6, txt=f"   - NEGATIF : Benar (TP)={tp_neg} | Salah sbg Negatif (FP)={fp_neg} | Gagal tebak Negatif (FN)={fn_neg}", ln=True)
                    pdf.ln(5)
                    
                    # 3. Metrik
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt="2. Hasil Evaluasi Metrik (Macro-Average):", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(0, 6, txt=f"   - Total Data Uji : {total_data} Baris", ln=True)
                    pdf.cell(0, 6, txt=f"   - Accuracy       : {acc*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   - Precision      : {precision*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   - Recall         : {recall*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   - F1-Score       : {f1*100:.2f}%", ln=True)
                    pdf.ln(5)
                    
                    # 4. Rumus (Dipersingkat agar muat di 1 halaman)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt="3. Pembuktian Rumus Evaluasi:", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(0, 6, txt=f"   A. Accuracy = (Total Prediksi Benar) / (Total Data Uji) = ({total_benar}) / ({total_data}) = {acc*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   B. Precision (Macro-Avg) = {precision*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   C. Recall (Macro-Avg) = {recall*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   D. F1-Score (Macro-Avg) = {f1*100:.2f}%", ln=True)
                    pdf.ln(5)

                    # 5. Kesimpulan
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt="4. Kesimpulan Evaluasi:", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.multi_cell(0, 6, txt=teks_kesimpulan)

                    # Generate PDF
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    
                    st.download_button(
                        label="📥 Download Full Laporan (PDF)",
                        data=pdf_bytes,
                        file_name="laporan_evaluasi_indobert_lengkap.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )