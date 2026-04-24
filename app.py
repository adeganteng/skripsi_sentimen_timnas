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
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import altair as alt

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
# ==========================================
# 3. SIDEBAR NAVIGASI (UI PREMIUM)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚽ Sentimen Timnas</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Membuat Menu Estetik
    menu = option_menu(
        menu_title=None,  # Kita kosongin karena udah ada judul di atas
        options=["Komentar Tunggal", "Analisis Dataset", "Evaluasi Model"],
        icons=["chat-quote", "bar-chart-line", "clipboard-data"], # Menggunakan Bootstrap Icons
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#ff4b4b", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "0px", 
                "--hover-color": "#f0f2f6"
            },
            "nav-link-selected": {"background-color": "#ff4b4b", "color": "white", "font-weight": "bold"},
        }
    )
    
    st.markdown("---")
    st.caption("© 2026 - Ade Wicaksono")

# ==========================================
# 4. LOGIKA HALAMAN MENU
# ==========================================

# ----------------------------------------------------
# MENU 1: ANALISIS KOMENTAR TUNGGAL
# ----------------------------------------------------
if menu == "Komentar Tunggal":
    st.title("🔍 Analisis Komentar Tunggal")
    st.write("Masukkan 1 komentar terkait Timnas Indonesia untuk dianalisis oleh model IndoBERT.")

    # Input Box dari user
    user_input = st.text_area("Masukkan Komentar:", placeholder="Contoh: Timnas mainnya bagus banget hari ini, bangga!")

    # Tombol Eksekusi
    if st.button("🚀 Analisis Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("Komentarnya jangan kosong dong, bro!")
        else:
            with st.spinner("Model sedang menganalisis pola kalimat..."):
                
                # ==========================================
                # A. PREPROCESSING BERTAHAP (UNTUK UI)
                # ==========================================
                raw_text = str(user_input)
                step_casefolding = raw_text.lower()
                step_cleansing1 = re.sub(r'@[A-Za-z0-9_]+|#\w+|http\S+|www\S+|https\S+', '', step_casefolding, flags=re.MULTILINE)
                step_cleansing2 = re.sub(r'[^a-z\s]', ' ', step_cleansing1)
                clean_text = re.sub(r'\s+', ' ', step_cleansing2).strip()

                # ==========================================
                # B. TOKENIZATION & PREDIKSI AI
                # ==========================================
                inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

                with torch.no_grad():
                    outputs = model(**inputs)
                    # Mengubah logit (skor mentah) menjadi probabilitas 0-100% menggunakan Softmax
                    probs = F.softmax(outputs.logits, dim=1).squeeze()
                    # TAMPILKAN INI BARU: Ambil skor mentah (logits) untuk perhitungan manual
                    logits = outputs.logits.squeeze().tolist()

                prob_positif = probs[0].item() * 100
                prob_netral = probs[1].item() * 100
                prob_negatif = probs[2].item() * 100
                
                z_pos, z_net, z_neg = logits[0], logits[1], logits[2]

                predicted_class_id = torch.argmax(probs).item()
                label_map = {0: "Positif", 1: "Netral", 2: "Negatif"}
                sentiment = label_map[predicted_class_id]
                confidence_score = max(prob_positif, prob_netral, prob_negatif)

                # ==========================================
                # C. DETEKSI KATA KUNCI (EXPLAINABILITY)
                # ==========================================
                # Dictionary sederhana untuk menangkap kata yang mempengaruhi model
                kata_positif = ['bagus', 'keren', 'menang', 'bangga', 'mantap', 'hebat', 'terbaik', 'top', 'berkembang']
                kata_negatif = ['jelek', 'kalah', 'kecewa', 'buruk', 'pecat', 'lemah', 'payah', 'bapuk', 'evaluasi']
                
                found_pos = [w for w in clean_text.split() if w in kata_positif]
                found_neg = [w for w in clean_text.split() if w in kata_negatif]

                # ==========================================
                # D. LOGIKA PENJELASAN DINAMIS (SUMMARY)
                # ==========================================
                if sentiment == "Positif":
                    alasan_kalimat = "Berdasarkan pola bahasa, model mendeteksi adanya kosakata yang menunjukkan dukungan, pujian, atau rasa bangga terhadap performa Timnas Indonesia."
                elif sentiment == "Negatif":
                    alasan_kalimat = "Model mengidentifikasi adanya pola kalimat yang mengandung kritik, kekecewaan, atau nada pesimis terhadap pemain maupun pelatih Timnas."
                else:
                    alasan_kalimat = "Model tidak menemukan kecenderungan emosi yang kuat. Kalimat ini diklasifikasikan sebagai netral karena sifatnya yang informatif atau ambigu."

                if confidence_score >= 90:
                    alasan_angka = f"Model **sangat yakin** dengan prediksi ini."
                elif confidence_score >= 70:
                    alasan_angka = f"Model **cukup yakin** dengan prediksi ini."
                else:
                    alasan_angka = f"Model **kurang yakin** (probabilitas terbagi rata)."

                penjelasan_lengkap = f"{alasan_kalimat} {alasan_angka}"

                # ==========================================
                # E. TAMPILAN OUTPUT (UI)
                # ==========================================
                st.markdown("---")
                st.subheader("📊 Hasil Analisis")

                # Membagi layout menjadi 2 kolom
                col1, col2 = st.columns([1.2, 1])
                
                with col1:
                    st.metric(label="Prediksi Sentimen Akhir", value=sentiment)
                    st.info(f"**💡 Mengapa hasilnya demikian?**\n\n{penjelasan_lengkap}")
                    
                    # Menampilkan kata kunci jika terdeteksi
                    if found_pos or found_neg:
                        st.write("**🔍 Kata Kunci yang Terdeteksi:**")
                        if found_pos: 
                            st.success(f"Mendorong ke arah Positif: {', '.join(found_pos)}")
                        if found_neg: 
                            st.error(f"Mendorong ke arah Negatif: {', '.join(found_neg)}")
                    
                with col2:
                    # Menentukan warna berdasarkan sentimen
                    warna_bar = "#95a5a6" # Default Abu-abu (Netral)
                    if sentiment == "Positif": warna_bar = "#2ecc71" # Hijau
                    elif sentiment == "Negatif": warna_bar = "#e74c3c" # Merah
                        
                    # Grafik Speedometer (Gauge)
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence_score,
                        title = {'text': "Keyakinan Model", 'font': {'size': 18}},
                        number = {'suffix': "%", 'valueformat': ".1f", 'font': {'size': 40, 'color': warna_bar}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': warna_bar},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': "#f0f2f6"},
                                {'range': [50, 80], 'color': "#e1e4e8"}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # --- VISUALISASI BAR CHART & PENJELASAN MATEMATIS ---
                st.markdown("### 📈 Detail Probabilitas Kelas")
                
                # Buat DataFrame untuk Bar Chart
                df_probs = pd.DataFrame({
                    'Sentimen': ['Positif', 'Netral', 'Negatif'],
                    'Persentase (%)': [prob_positif, prob_netral, prob_negatif],
                    'Warna': ['#2ecc71', '#95a5a6', '#e74c3c']
                })

                # Visualisasi menggunakan Altair
                chart = alt.Chart(df_probs).mark_bar(cornerRadiusEnd=4, height=30).encode(
                    x=alt.X('Persentase (%):Q', scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('Sentimen:N', sort=None, title=""),
                    color=alt.Color('Warna:N', scale=None),
                    tooltip=['Sentimen', alt.Tooltip('Persentase (%):Q', format='.2f')]
                ).properties(height=150)
                
                st.altair_chart(chart, use_container_width=True)

                # --- EXPANDERS (DI BALIK LAYAR) ---
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    with st.expander("🛠️ Tahapan Preprocessing NLP"):
                        st.markdown("**1. Teks Asli**")
                        st.code(raw_text, language="text")
                        st.markdown("**2. Case Folding**")
                        st.code(step_casefolding, language="text")
                        st.markdown("**3. Cleansing (URL, Mention, Hashtag)**")
                        st.code(step_cleansing1, language="text")
                        st.markdown("**4. Cleansing (Simbol & Angka)**")
                        st.code(step_cleansing2, language="text")
                        st.markdown("**5. Teks Bersih Akhir**")
                        st.code(clean_text, language="text")

                with col_exp2:
                    with st.expander("🤔 Kok Bisa Dapat Angka Tersebut? (Detail Matematika)"):
                        st.write("Persentase keyakinan didapatkan menggunakan fungsi matematis bernama **Softmax**. Fungsi ini mengubah skor mentah (*logits*) dari AI menjadi probabilitas berskala 0 hingga 100%.")
                        
                        # 1. Rumus Umum
                        st.latex(r"P(i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}")
                        
                        st.markdown(r"""
                        **Keterangan:**
                        *   $P(i)$ = Probabilitas hasil akhir (0-1)
                        *   $z_i$ = Skor mentah (*logit*) dari model
                        *   $e$ = Bilangan Euler ($\approx 2.718$)
                        """)
                        
                        st.divider()
                        
                        # 2. Nilai Logits Mentah
                        st.markdown("**1️⃣ Skor Mentah (Logits) dari Model:**")
                        st.write(f"- $z_{{positif}}$ = `{z_pos:.4f}`")
                        st.write(f"- $z_{{netral}}$ = `{z_net:.4f}`")
                        st.write(f"- $z_{{negatif}}$ = `{z_neg:.4f}`")
                        
                        # 3. Substitusi ke Rumus
                        st.markdown(f"**2️⃣ Memasukkan ke Rumus Softmax (fokus ke {sentiment}):**")
                        
                        # Mengambil nilai logit dari sentimen yang menang
                        z_menang = logits[predicted_class_id]
                        
                        # Membuat string LaTeX dinamis untuk rumus
                        str_num = f"e^{{{z_menang:.2f}}}"
                        str_den = f"e^{{{z_pos:.2f}}} + e^{{{z_net:.2f}}} + e^{{{z_neg:.2f}}}"
                        st.latex(rf"P(\text{{{sentiment}}}) = \frac{{{str_num}}}{{{str_den}}}")
                        
                        # 4. Hasil Eksponensial (e^z)
                        import math
                        exp_pos = math.exp(z_pos)
                        exp_net = math.exp(z_net)
                        exp_neg = math.exp(z_neg)
                        total_exp = exp_pos + exp_net + exp_neg
                        exp_menang = math.exp(z_menang)
                        
                        st.markdown("**3️⃣ Menghitung Nilai Eksponensial ($e^z$):**")
                        st.latex(rf"P(\text{{{sentiment}}}) = \frac{{{exp_menang:.2f}}}{{{exp_pos:.2f} + {exp_net:.2f} + {exp_neg:.2f}}}")
                        
                        # 5. Hasil Pembagian Akhir
                        prob_final = exp_menang / total_exp
                        st.markdown("**4️⃣ Hasil Akhir:**")
                        st.latex(rf"P(\text{{{sentiment}}}) = \frac{{{exp_menang:.2f}}}{{{total_exp:.2f}}} = {prob_final:.4f}")
                        
                        st.info(f"Karena $P({sentiment})$ adalah **{prob_final:.4f}**, jika dikalikan 100%, kita mendapatkan persentase final yaitu **{prob_final * 100:.2f}%**!")
                        
# ----------------------------------------------------
# MENU 2: ANALISIS DATASET (TANPA LABEL)
# ----------------------------------------------------
elif menu == "Analisis Dataset":
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
            
            # ==========================================
            # FITUR BARU: EDUKASI PREPROCESSING DATASET
            # ==========================================
            st.markdown("---")
            st.subheader("🛠️ Di Balik Layar: Simulasi Preprocessing Data")
            st.write("Sebelum model AI bekerja secara massal, sistem akan membersihkan teks dari simbol, link, dan emoji. Berikut adalah simulasi tahapan pembersihan pada 5 baris pertama dataset Anda:")
            
            with st.expander("🔍 Buka Tabel Tahapan Preprocessing (Top 5 Data)"):
                # Kita bikin dataframe tiruan khusus untuk pamer proses step-by-step
                df_preview = df.head(5).copy()
                df_preview['1. Teks Asli'] = df_preview['komentar'].astype(str)
                df_preview['2. Case Folding'] = df_preview['1. Teks Asli'].str.lower()
                df_preview['3. Hapus Link/Tag'] = df_preview['2. Case Folding'].apply(lambda x: re.sub(r'@[A-Za-z0-9_]+|#\w+|http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
                df_preview['4. Hapus Simbol'] = df_preview['3. Hapus Link/Tag'].apply(lambda x: re.sub(r'[^a-z\s]', ' ', x))
                df_preview['5. Teks Bersih Akhir'] = df_preview['4. Hapus Simbol'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
                
                st.dataframe(df_preview[['1. Teks Asli', '2. Case Folding', '3. Hapus Link/Tag', '4. Hapus Simbol', '5. Teks Bersih Akhir']])
                st.caption("Proses ini otomatis diterapkan pada seluruh baris data di belakang layar agar model dapat fokus menganalisis emosi dari kata dasarnya saja.")

            # ==========================================
            # PROSES ANALISIS & PREDIKSI
            # ==========================================
            if st.button("🚀 Mulai Analisis Massal & Buat Laporan", type="primary"):
                
                progress_text = "AI sedang membaca, memprediksi, dan mengkalkulasi probabilitas. Mohon tunggu..."
                my_bar = st.progress(0, text=progress_text)
                
                sentimens = []
                label_ints = [] 
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
                    label_ints.append(pred_id) 
                    prob_pos_list.append(f"{p_pos:.1f}%")
                    prob_net_list.append(f"{p_net:.1f}%")
                    prob_neg_list.append(f"{p_neg:.1f}%")
                    
                    my_bar.progress((i + 1) / len(df), text=f"Memproses baris {i+1} dari {len(df)}...")
                
                my_bar.empty()
                
                # Masukkan list ke dalam DataFrame
                df['Prediksi_Sentimen'] = sentimens
                df['label'] = label_ints 
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
                
                # Header Tabel PDF
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
                    st.download_button("📥 Laporan PDF Lengkap", data=pdf_bytes, file_name='Laporan_Analisis_Timnas.pdf', mime='application/pdf', type='primary', width='stretch')
                with col_dl2: # Tombol CSV
                    csv_hasil = df[kolom_tampil].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Data CSV (.csv)", data=csv_hasil, file_name='dataset_timnas_berlabel.csv', mime='text/csv', width='stretch')
                with col_dl3: # Tombol Excel
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df[kolom_tampil].to_excel(writer, index=False, sheet_name='Hasil Sentimen')
                    st.download_button("📥 Data Excel (.xlsx)", data=excel_buffer.getvalue(), file_name='dataset_timnas_berlabel.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', width='stretch')

# ----------------------------------------------------
# MENU 3: EVALUASI PERFORMA MODEL
# ----------------------------------------------------
elif menu == "Evaluasi Model":
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
            
            # ==========================================
            # FITUR BARU: EDUKASI PREPROCESSING DATA UJI
            # ==========================================
            st.markdown("---")
            st.subheader("🛠️ Di Balik Layar: Simulasi Preprocessing Data Uji")
            st.write("Sesuai standar pengujian Machine Learning, data uji (Testing Data) juga harus melewati tahap pembersihan yang sama persis dengan data latih. Berikut simulasinya:")
            
            with st.expander("🔍 Buka Tabel Tahapan Preprocessing (Top 5 Data Uji)"):
                df_preview_test = df_test.head(5).copy()
                df_preview_test['1. Teks Asli'] = df_preview_test['komentar'].astype(str)
                df_preview_test['2. Case Folding'] = df_preview_test['1. Teks Asli'].str.lower()
                df_preview_test['3. Hapus Link/Tag'] = df_preview_test['2. Case Folding'].apply(lambda x: re.sub(r'@[A-Za-z0-9_]+|#\w+|http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
                df_preview_test['4. Hapus Simbol'] = df_preview_test['3. Hapus Link/Tag'].apply(lambda x: re.sub(r'[^a-z\s]', ' ', x))
                df_preview_test['5. Teks Bersih Akhir'] = df_preview_test['4. Hapus Simbol'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
                
                st.dataframe(df_preview_test[['label', '1. Teks Asli', '2. Case Folding', '3. Hapus Link/Tag', '4. Hapus Simbol', '5. Teks Bersih Akhir']])
                st.caption("Label asli tetap dipertahankan, sementara teks dibersihkan agar model dapat menebak secara adil tanpa terpengaruh noise.")

            # ==========================================
            # PROSES EVALUASI & METRIK
            # ==========================================
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

                    # Kalkulasi Metrik & Matriks (OvR & Gabungan)
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                    total_data = len(df_test)
                    
                    # Ekstrak OvR (One-vs-Rest)
                    # POSITIF
                    tp_pos = cm[0][0]
                    fn_pos = cm[0][1] + cm[0][2]
                    fp_pos = cm[1][0] + cm[2][0]
                    tn_pos = total_data - (tp_pos + fn_pos + fp_pos)
                    prec_pos = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0
                    rec_pos = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0
                    f1_pos = 2 * (prec_pos * rec_pos) / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0

                    # NETRAL
                    tp_net = cm[1][1]
                    fn_net = cm[1][0] + cm[1][2]
                    fp_net = cm[0][1] + cm[2][1]
                    tn_net = total_data - (tp_net + fn_net + fp_net)
                    prec_net = tp_net / (tp_net + fp_net) if (tp_net + fp_net) > 0 else 0
                    rec_net = tp_net / (tp_net + fn_net) if (tp_net + fn_net) > 0 else 0
                    f1_net = 2 * (prec_net * rec_net) / (prec_net + rec_net) if (prec_net + rec_net) > 0 else 0

                    # NEGATIF
                    tp_neg = cm[2][2]
                    fn_neg = cm[2][0] + cm[2][1]
                    fp_neg = cm[0][2] + cm[1][2]
                    tn_neg = total_data - (tp_neg + fn_neg + fp_neg)
                    prec_neg = tp_neg / (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else 0
                    rec_neg = tp_neg / (tp_neg + fn_neg) if (tp_neg + fn_neg) > 0 else 0
                    f1_neg = 2 * (prec_neg * rec_neg) / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0

                    # METRIK GABUNGAN (MACRO)
                    total_benar = tp_pos + tp_net + tp_neg
                    acc = accuracy_score(y_true, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

                    st.markdown("---")
                    st.header("🎯 Laporan Evaluasi Terperinci (One-vs-Rest)")
                    
                    # TABS NAVIGASI
                    tab_pos, tab_net, tab_neg, tab_gabungan = st.tabs([
                        "🟢 Khusus Positif", "⚪ Khusus Netral", "🔴 Khusus Negatif", "🔵 Gabungan (Keseluruhan)"
                    ])

                    # TAB 1: POSITIF
                    with tab_pos:
                        st.subheader("Evaluasi Kelas: Sentimen Positif")
                        col_gbr, col_rumus = st.columns([1, 1.5])
                        
                        with col_gbr:
                            fig_pos, ax_pos = plt.subplots(figsize=(4, 3))
                            sns.heatmap([[tn_pos, fp_pos], [fn_pos, tp_pos]], annot=True, fmt='d', cmap='Greens',
                                        xticklabels=['Bukan Positif', 'Positif'], yticklabels=['Bukan Positif', 'Positif'], ax=ax_pos)
                            ax_pos.set_ylabel('Aktual Asli')
                            ax_pos.set_xlabel('Prediksi Model')
                            st.pyplot(fig_pos)
                            fig_pos.savefig("temp_cm_pos.png", bbox_inches='tight') 
                            
                        with col_rumus:
                            st.write(f"**Rincian Data:** Benar (TP): {tp_pos} | Salah Tebak (FP): {fp_pos} | Gagal Tebak (FN): {fn_pos}")
                            st.latex(rf"Precision = \frac{{TP}}{{TP + FP}} = \frac{{{tp_pos}}}{{{tp_pos} + {fp_pos}}} = {prec_pos*100:.1f}\%")
                            st.latex(rf"Recall = \frac{{TP}}{{TP + FN}} = \frac{{{tp_pos}}}{{{tp_pos} + {fn_pos}}} = {rec_pos*100:.1f}\%")
                            st.latex(rf"F1-Score = 2 \times \frac{{Prec \times Rec}}{{Prec + Rec}} = {f1_pos*100:.1f}\%")

                    # TAB 2: NETRAL
                    with tab_net:
                        st.subheader("Evaluasi Kelas: Sentimen Netral")
                        col_gbr, col_rumus = st.columns([1, 1.5])
                        
                        with col_gbr:
                            fig_net, ax_net = plt.subplots(figsize=(4, 3))
                            sns.heatmap([[tn_net, fp_net], [fn_net, tp_net]], annot=True, fmt='d', cmap='Greys',
                                        xticklabels=['Bukan Netral', 'Netral'], yticklabels=['Bukan Netral', 'Netral'], ax=ax_net)
                            ax_net.set_ylabel('Aktual Asli')
                            ax_net.set_xlabel('Prediksi Model')
                            st.pyplot(fig_net)
                            fig_net.savefig("temp_cm_net.png", bbox_inches='tight')
                            
                        with col_rumus:
                            st.write(f"**Rincian Data:** Benar (TP): {tp_net} | Salah Tebak (FP): {fp_net} | Gagal Tebak (FN): {fn_net}")
                            st.latex(rf"Precision = \frac{{TP}}{{TP + FP}} = \frac{{{tp_net}}}{{{tp_net} + {fp_net}}} = {prec_net*100:.1f}\%")
                            st.latex(rf"Recall = \frac{{TP}}{{TP + FN}} = \frac{{{tp_net}}}{{{tp_net} + {fn_net}}} = {rec_net*100:.1f}\%")
                            st.latex(rf"F1-Score = 2 \times \frac{{Prec \times Rec}}{{Prec + Rec}} = {f1_net*100:.1f}\%")

                    # TAB 3: NEGATIF
                    with tab_neg:
                        st.subheader("Evaluasi Kelas: Sentimen Negatif")
                        col_gbr, col_rumus = st.columns([1, 1.5])
                        
                        with col_gbr:
                            fig_neg, ax_neg = plt.subplots(figsize=(4, 3))
                            sns.heatmap([[tn_neg, fp_neg], [fn_neg, tp_neg]], annot=True, fmt='d', cmap='Reds',
                                        xticklabels=['Bukan Negatif', 'Negatif'], yticklabels=['Bukan Negatif', 'Negatif'], ax=ax_neg)
                            ax_neg.set_ylabel('Aktual Asli')
                            ax_neg.set_xlabel('Prediksi Model')
                            st.pyplot(fig_neg)
                            fig_neg.savefig("temp_cm_neg.png", bbox_inches='tight')
                            
                        with col_rumus:
                            st.write(f"**Rincian Data:** Benar (TP): {tp_neg} | Salah Tebak (FP): {fp_neg} | Gagal Tebak (FN): {fn_neg}")
                            st.latex(rf"Precision = \frac{{TP}}{{TP + FP}} = \frac{{{tp_neg}}}{{{tp_neg} + {fp_neg}}} = {prec_neg*100:.1f}\%")
                            st.latex(rf"Recall = \frac{{TP}}{{TP + FN}} = \frac{{{tp_neg}}}{{{tp_neg} + {fn_neg}}} = {rec_neg*100:.1f}\%")
                            st.latex(rf"F1-Score = 2 \times \frac{{Prec \times Rec}}{{Prec + Rec}} = {f1_neg*100:.1f}\%")

                    # TAB 4: GABUNGAN
                    with tab_gabungan:
                        st.subheader("🧩 Confusion Matrix Gabungan & Hasil Akhir")
                        
                        col_gab1, col_gab2 = st.columns([1, 1.2])
                        with col_gab1:
                            label_names = ['Positif', 'Netral', 'Negatif']
                            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax_cm)
                            ax_cm.set_ylabel('Label Aktual (Asli)')
                            ax_cm.set_xlabel('Label Prediksi (Model)')
                            st.pyplot(fig_cm)
                            fig_cm.savefig("temp_cm_gab.png", bbox_inches='tight')
                        
                        with col_gab2:
                            st.write("**Akurasi Keseluruhan (Accuracy):**")
                            st.latex(rf"Accuracy = \frac{{{total_benar} \text{{ (Benar)}} }}{{{total_data} \text{{ (Total Data)}}}} = {acc*100:.1f}\%")
                            st.write("**Rata-Rata Makro (Macro-Average):**")
                            st.info(f"- **Precision Macro:** {precision*100:.1f}%\n- **Recall Macro:** {recall*100:.1f}%\n- **F1-Score Macro:** {f1*100:.1f}%")
                            
                            kategori_performa = "sangat baik" if acc >= 0.85 else "cukup baik" if acc >= 0.75 else "masih kurang"
                            teks_kesimpulan = f"Secara keseluruhan, model mencapai akurasi **{acc*100:.1f}%** ({kategori_performa}). Rincian masing-masing kelas membuktikan bahwa model dapat membedakan opini suporter Timnas dengan proporsional."
                            st.success(teks_kesimpulan)

                    # ==========================================
                    # FITUR EXPORT LAPORAN PDF
                    # ==========================================
                    st.markdown("---")
                    st.subheader("📄 Cetak Laporan Evaluasi Full (PDF)")
                    
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Judul
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, txt="Laporan Evaluasi Performa Model IndoBERT", ln=True, align='C')
                    pdf.set_font("Arial", size=11)
                    pdf.cell(0, 8, txt="Analisis Sentimen Komentar Instagram @timnasindonesia", ln=True, align='C')
                    pdf.ln(5)
                    
                    # 1. Gambar CM Individual
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt="1. Visualisasi Confusion Matrix Per Kelas (One-vs-Rest):", ln=True)
                    
                    y_pos = pdf.get_y()
                    pdf.image("temp_cm_pos.png", x=10, y=y_pos, w=60)
                    pdf.image("temp_cm_net.png", x=75, y=y_pos, w=60)
                    pdf.image("temp_cm_neg.png", x=140, y=y_pos, w=60)
                    pdf.ln(48)
                    
                    # 2. Ringkasan Metrik Individual
                    pdf.set_font("Arial", 'B', 11)
                    pdf.cell(0, 8, txt="   Ringkasan Metrik Masing-Masing Kelas:", ln=True)
                    pdf.set_font("Arial", size=10)
                    pdf.cell(0, 6, txt=f"   - POSITIF : Prec={prec_pos*100:.1f}% | Rec={rec_pos*100:.1f}% | F1={f1_pos*100:.1f}%  (TP={tp_pos}, FP={fp_pos}, FN={fn_pos})", ln=True)
                    pdf.cell(0, 6, txt=f"   - NETRAL  : Prec={prec_net*100:.1f}% | Rec={rec_net*100:.1f}% | F1={f1_net*100:.1f}%  (TP={tp_net}, FP={fp_net}, FN={fn_net})", ln=True)
                    pdf.cell(0, 6, txt=f"   - NEGATIF : Prec={prec_neg*100:.1f}% | Rec={rec_neg*100:.1f}% | F1={f1_neg*100:.1f}%  (TP={tp_neg}, FP={fp_neg}, FN={fn_neg})", ln=True)
                    pdf.ln(5)
                    
                    # 3. Gambar CM Gabungan
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt="2. Confusion Matrix Gabungan (Multiclass):", ln=True)
                    pdf.image("temp_cm_gab.png", x=60, w=90)
                    pdf.ln(5)
                    
                    # 4. Metrik Gabungan
                    pdf.cell(0, 10, txt="3. Metrik Evaluasi Akhir (Macro-Average):", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(0, 6, txt=f"   - Total Data Uji : {total_data} Baris", ln=True)
                    pdf.cell(0, 6, txt=f"   - Akurasi (Accuracy)  = ({total_benar}) / ({total_data}) = {acc*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   - Precision (Macro)   = {precision*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   - Recall (Macro)      = {recall*100:.2f}%", ln=True)
                    pdf.cell(0, 6, txt=f"   - F1-Score (Macro)    = {f1*100:.2f}%", ln=True)
                    pdf.ln(5)
                    
                    # 5. Kesimpulan
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt="4. Kesimpulan Evaluasi:", ln=True)
                    pdf.set_font("Arial", size=11)
                    
                    teks_bersih_pdf = str(teks_kesimpulan).encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 6, txt=teks_bersih_pdf)

                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    
                    st.download_button(
                        label="📥 Download Full Laporan (PDF)",
                        data=pdf_bytes,
                        file_name="laporan_evaluasi_indobert_ovr_lengkap.pdf",
                        mime="application/pdf",
                        type="primary",
                        width="stretch" 
                    )