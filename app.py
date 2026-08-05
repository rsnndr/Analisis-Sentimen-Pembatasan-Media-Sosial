import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle
from wordcloud import WordCloud
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Analisis Sentimen Pembatasan Media Sosial",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PALET WARNA PASTEL+
PASTEL_BLUE        = "#60A5FA"
PASTEL_BLUE_LIGHT  = "#DBEAFE"
PASTEL_GREEN       = "#34D399"
PASTEL_GREEN_LIGHT = "#DCFCE7"
PASTEL_PINK        = "#FB7185"
PASTEL_PINK_LIGHT  = "#FFE4E6"
PASTEL_LAVENDER    = "#A78BFA"
INK                = "#1E293B"
INK_SOFT           = "#475569"

COLOR_MAP  = {'positive': PASTEL_GREEN, 'negative': PASTEL_PINK, 'neutral': PASTEL_BLUE}
EMOJI_MAP  = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
LABEL_MAP  = {'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Netral'}

# CUSTOM CSS 
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{ background: #F8FBFF; }}

.main-header {{
    background: linear-gradient(135deg, {PASTEL_BLUE_LIGHT} 0%, {PASTEL_GREEN_LIGHT} 100%);
    padding: 2rem 2rem; border-radius: 20px; margin-bottom: 1.5rem;
    border: 1px solid #EAF2FB;
}}
.main-header h1 {{ font-size: 1.5rem; font-weight: 700; margin: 0 0 0.4rem 0; color: {INK}; }}
.main-header p  {{ font-size: 0.88rem; color: {INK_SOFT}; margin: 0; }}

.metric-card {{
    background: white; border: 1px solid #E7EFFA; border-radius: 14px;
    padding: 1rem 1.2rem; text-align: center; box-shadow: 0 4px 14px rgba(96,165,250,0.10);
}}
.metric-card .metric-value {{ font-size: 1.8rem; font-weight: 700; line-height: 1; margin-bottom: 0.3rem; }}
.metric-card .metric-label {{ font-size: 0.72rem; color: #7C8CA6; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }}
.metric-card .metric-sub   {{ font-size: 0.7rem; color: #A9B7CC; margin-top: 0.2rem; }}

.section-header {{
    font-size: 1rem; font-weight: 600; color: {INK};
    margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid {PASTEL_BLUE_LIGHT};
}}
.badge-pos {{ background:{PASTEL_GREEN_LIGHT}; color:#047857; border:1px solid #6EE7B7; padding:0.2rem 0.7rem; border-radius:999px; font-size:0.8rem; font-weight:600; }}
.badge-neg {{ background:{PASTEL_PINK_LIGHT}; color:#BE123C; border:1px solid #FDA4AF; padding:0.2rem 0.7rem; border-radius:999px; font-size:0.8rem; font-weight:600; }}
.badge-neu {{ background:{PASTEL_BLUE_LIGHT}; color:#1D4ED8; border:1px solid #93C5FD; padding:0.2rem 0.7rem; border-radius:999px; font-size:0.8rem; font-weight:600; }}

.stTabs [data-baseweb="tab-list"] {{ gap:8px; background:#EFF5FC; padding:6px; border-radius:12px; }}
.stTabs [data-baseweb="tab"]      {{ border-radius:10px; padding:0.5rem 1.2rem; font-weight:500; font-size:0.88rem; }}
.stTabs [aria-selected="true"]    {{ background:white !important; box-shadow:0 2px 6px rgba(96,165,250,0.18); }}

.acc-highlight {{ border-radius:14px; padding:0.9rem 1rem; text-align:center; margin-bottom:1rem; color:white; box-shadow: 0 6px 16px rgba(96,165,250,0.20); }}
.acc-highlight .acc-val   {{ font-size:1.9rem; font-weight:700; }}
.acc-highlight .acc-label {{ font-size:0.75rem; opacity:0.92; text-transform:uppercase; letter-spacing:0.05em; }}

.pred-card {{
    border-radius: 16px; padding: 1.5rem; text-align: center;
    border: 1px solid #E7EFFA; background: white; box-shadow: 0 4px 14px rgba(96,165,250,0.10);
}}
.pred-card .pred-emoji  {{ font-size: 3rem; line-height: 1; margin-bottom: 0.5rem; }}
.pred-card .pred-label  {{ font-size: 1.4rem; font-weight: 700; }}
.pred-card .pred-model  {{ font-size: 0.75rem; color: #7C8CA6; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.3rem; }}

.divider {{ border:none; border-top:1px solid {PASTEL_BLUE_LIGHT}; margin:1.5rem 0; }}
.footer  {{ text-align:center; color:#A9B7CC; font-size:0.78rem; margin-top:2rem; padding-top:1rem; border-top:1px solid {PASTEL_BLUE_LIGHT}; }}
</style>
""", unsafe_allow_html=True)


# HELPER: GRAFIK BERGAYA GRADIENT
def gradient_vbar(ax, x, height, width, color_hex, bottom=0):
    """Bar vertikal dengan isian gradient dari putih ke warna target."""
    if height <= 0:
        return
    grad = np.linspace(0, 1, 256).reshape(256, 1)
    cmap = LinearSegmentedColormap.from_list('g', ['#FFFFFF', color_hex])
    extent = [x - width/2, x + width/2, bottom, bottom + height]
    ax.imshow(grad, extent=extent, aspect='auto', cmap=cmap, origin='lower',
              zorder=2, interpolation='bilinear')
    ax.add_patch(Rectangle((x - width/2, bottom), width, height,
                            fill=False, edgecolor='white', linewidth=2, zorder=3))

def gradient_hbar(ax, y, width, height, color_hex, left=0):
    """Bar horizontal dengan isian gradient dari putih ke warna target."""
    if width <= 0:
        return
    grad = np.linspace(0, 1, 256).reshape(1, 256)
    cmap = LinearSegmentedColormap.from_list('g', ['#FFFFFF', color_hex])
    extent = [left, left + width, y - height/2, y + height/2]
    ax.imshow(grad, extent=extent, aspect='auto', cmap=cmap, origin='lower',
              zorder=2, interpolation='bilinear')
    ax.add_patch(Rectangle((left, y - height/2), width, height,
                            fill=False, edgecolor='white', linewidth=2, zorder=3))

PASTEL_CMAP = LinearSegmentedColormap.from_list('pastel_blue_green',
                                                 ['#FFFFFF', PASTEL_BLUE_LIGHT, PASTEL_BLUE])


# LEXICON (InSet)
import requests, csv
from io import StringIO

@st.cache_resource(show_spinner=False)
def load_lexicon():
    lex_pos, lex_neg = {}, {}
    try:
        r = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv', timeout=10)
        if r.status_code == 200:
            for row in csv.reader(StringIO(r.text)):
                if len(row) >= 2:
                    try: lex_pos[row[0]] = int(row[1])
                    except: pass
        r = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv', timeout=10)
        if r.status_code == 200:
            for row in csv.reader(StringIO(r.text)):
                if len(row) >= 2:
                    try: lex_neg[row[0]] = int(row[1])
                    except: pass
    except:
        pass
    return lex_pos, lex_neg

def predict_lexicon(text_tokens, lex_pos, lex_neg):
    score = sum(lex_pos.get(w, 0) for w in text_tokens) + sum(lex_neg.get(w, 0) for w in text_tokens)
    if score > 0:   return score, 'positive'
    elif score < 0: return score, 'negative'
    else:           return score, 'neutral'

def simple_preprocess(text):
    text = re.sub(r'@\S+|#\S+|http\S+|RT[\s]', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    return text


# LOAD DATA & TRAIN
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("hasil_preprocessing.csv")

@st.cache_resource(show_spinner=False)
def train_models(df_hash):
    df = load_data()
    X = df['text_akhir'].fillna('')
    y = df['polarity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    tfidf = TfidfVectorizer()
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)
    svm = SVC(kernel='linear', random_state=42, probability=True)
    svm.fit(Xtr, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(Xtr, y_train)
    return tfidf, svm, rf, y_test, svm.predict(Xte), rf.predict(Xte)


# INIT
try:
    clean_df               = load_data()
    tfidf, svm_model, rf_model, y_test, y_pred_svm, y_pred_rf = train_models(len(clean_df))
    lex_pos, lex_neg       = load_lexicon()
    data_ok                = True
except Exception as e:
    data_ok = False

# SIDEBAR
st.sidebar.markdown("## 💬 Analisis Sentimen")
st.sidebar.caption("Pembatasan Media Sosial Anak < 16 Tahun")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi",
    ["🏠  Beranda", "📊  Distribusi Sentimen", "🤖  Evaluasi Model", "☁️  Word Cloud", "🔍  Coba Prediksi"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
if data_ok:
    acc_svm = accuracy_score(y_test, y_pred_svm)
    acc_rf  = accuracy_score(y_test, y_pred_rf)
    st.sidebar.success(f"✅ Dataset: {len(clean_df):,} komentar")
    st.sidebar.info(f"🤖 SVM: **{acc_svm*100:.1f}%**\n\n🌲 RF: **{acc_rf*100:.1f}%**")
else:
    st.sidebar.error("❌ Data tidak ditemukan")
    st.sidebar.caption("Pastikan `hasil_preprocessing.csv` ada di folder yang sama.")

st.sidebar.markdown("---")
st.sidebar.caption("SVM · Random Forest · TF-IDF · InSet Lexicon · CRISP-DM")


if not data_ok:
    st.error("Gagal memuat data. Pastikan file `hasil_preprocessing.csv` ada di folder yang sama dengan `app.py`.")
    st.stop()

# SHARED VARS
sentimen_counts = clean_df['polarity'].value_counts()
total_data = len(clean_df)
jml_pos    = sentimen_counts.get('positive', 0)
jml_neg    = sentimen_counts.get('negative', 0)
jml_neu    = sentimen_counts.get('neutral',  0)
acc_svm    = accuracy_score(y_test, y_pred_svm)
acc_rf     = accuracy_score(y_test, y_pred_rf)

# PAGE: BERANDA
if "Beranda" in page:

    st.markdown("""
    <div class="main-header">
        <h1>💬 Analisis Sentimen Publik</h1>
        <p>Pembatasan Media Sosial bagi Anak di Bawah 16 Tahun</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
    (c1, f"{total_data:,}", "Total Komentar", "Dataset", "#2d3748"),
    (c2, f"{jml_pos:,}", "Sentimen Positif",
     f"{jml_pos/total_data*100:.1f}% dari total", "#276749"),
    (c3, f"{jml_neu:,}", "Sentimen Netral",
     f"{jml_neu/total_data*100:.1f}% dari total", "#2c5282"),
    (c4, f"{jml_neg:,}", "Sentimen Negatif",
     f"{jml_neg/total_data*100:.1f}% dari total", "#9b2c2c"),
    (c5, f"{acc_svm*100:.1f}%", "Akurasi SVM",
     "Kernel Linear", "#553c9a"),
    (c6, f"{acc_rf*100:.1f}%", "Akurasi RF",
     "100 Pohon", "#2b6cb0"),
    ]
    for col, val, label, sub, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("### Tentang Aplikasi")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **📌 Tujuan**
        Mengklasifikasikan opini publik terhadap isu pembatasan media sosial bagi anak di bawah 16 tahun
        ke dalam tiga kelas sentimen: **Positif**, **Negatif**, dan **Netral**.
        """)
    with col_b:
        st.markdown("""
        **⚙️ Metodologi**
        Penelitian menggunakan kerangka **CRISP-DM** dengan pelabelan otomatis menggunakan
        **InSet Lexicon** dan klasifikasi menggunakan **SVM** serta **Random Forest** berbasis **TF-IDF**.
        """)
    with col_c:
        st.markdown("""
        **📦 Data**
        **📦 Data**
        Dataset terdiri dari **25.680 komentar** yang telah melalui proses preprocessing
        dan pelabelan sentimen menggunakan metode InSet Lexicon.
        """)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### Alur Sistem (CRISP-DM)")
    st.markdown("""
    ```
    Pengumpulan Data → Pra-pemrosesan Teks → Pelabelan (InSet Lexicon)
         → Ekstraksi Fitur (TF-IDF) → Klasifikasi (SVM & Random Forest)
              → Evaluasi Model → Deployment (Streamlit)
    ```
    """)

# PAGE: DISTRIBUSI SENTIMEN
elif "Distribusi" in page:

    st.markdown("""
    <div class="main-header">
        <h1>📊 Distribusi Sentimen</h1>
        <p>Sebaran opini publik dari seluruh komentar yang dikumpulkan</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Distribusi Sentimen Keseluruhan</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])

    labels_order = [k for k in ['negative', 'neutral', 'positive'] if k in sentimen_counts.index]
    vals   = [sentimen_counts[k] for k in labels_order]
    colors = [COLOR_MAP[k] for k in labels_order]

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#F8FBFF'); ax.set_facecolor('#F8FBFF')
        for i, (v, c) in enumerate(zip(vals, colors)):
            gradient_vbar(ax, x=i, height=v, width=0.55, color_hex=c)
            ax.text(i, v + max(vals) * 0.03, f'{v:,}', ha='center', va='bottom',
                    fontsize=11, fontweight='700', color=INK)
        ax.set_xlim(-0.6, len(labels_order) - 0.4)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_xticks(range(len(labels_order)))
        ax.set_xticklabels(labels_order)
        ax.set_xlabel('Sentimen', fontsize=10, color=INK_SOFT)
        ax.set_ylabel('Jumlah Komentar', fontsize=10, color=INK_SOFT)
        ax.tick_params(colors=INK_SOFT)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#E2E8F0')
        ax.yaxis.grid(True, color='#EAF1FB', linewidth=1); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#F8FBFF'); ax2.set_facecolor('#F8FBFF')
        wedges, texts, autotexts = ax2.pie(
            vals, labels=labels_order, autopct='%1.1f%%',
            colors=colors, startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=3, width=0.42), pctdistance=0.80
        )
        for t in texts:    t.set_fontsize(10); t.set_color(INK); t.set_fontweight('600')
        for a in autotexts: a.set_fontsize(9);  a.set_color('white');   a.set_fontweight('700')
        # Donut center: total komentar
        ax2.text(0, 0.08, f'{total_data:,}', ha='center', va='center',
                  fontsize=15, fontweight='800', color=INK)
        ax2.text(0, -0.12, 'komentar', ha='center', va='center',
                  fontsize=8.5, color=INK_SOFT)
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# PAGE: EVALUASI MODEL
elif "Evaluasi" in page:

    st.markdown("""
    <div class="main-header">
        <h1>🤖 Evaluasi Model</h1>
        <p>Perbandingan performa SVM dan Random Forest pada data uji</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Perbandingan Performa Model</div>", unsafe_allow_html=True)
    col_svm, col_rf = st.columns(2)

    def render_model_card(col, nama, y_pred, grad, cm_cmap):
        with col:
            acc    = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.markdown(f"""
            <div class="acc-highlight" style="background:linear-gradient(135deg,{grad[0]},{grad[1]});">
                <div class="acc-val">{acc*100:.2f}%</div>
                <div class="acc-label">Akurasi {nama}</div>
            </div>""", unsafe_allow_html=True)

            kelas_list = [k for k in ['negative','neutral','positive','macro avg'] if k in report]
            cols_m = st.columns(len(kelas_list))
            for cm, kelas in zip(cols_m, kelas_list):
                with cm:
                    st.markdown(f"""
                    <div style="background:#F7FAFF;border:1px solid #E7EFFA;border-radius:10px;
                                padding:0.5rem;text-align:center;margin-bottom:0.5rem">
                        <div style="font-size:0.65rem;color:#7C8CA6;text-transform:uppercase;
                                    letter-spacing:0.04em;font-weight:500">{kelas.replace(' avg','').title()}</div>
                        <div style="font-size:0.95rem;font-weight:700;color:{INK}">
                            {report[kelas]['f1-score']:.2f}</div>
                        <div style="font-size:0.6rem;color:#A9B7CC">F1-score</div>
                    </div>""", unsafe_allow_html=True)

            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            fig_cm.patch.set_facecolor('#F8FBFF')
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap=cm_cmap, colorbar=False)
            ax_cm.set_title(f'Confusion Matrix — {nama}', fontsize=10, color=INK, pad=10)
            ax_cm.tick_params(colors=INK_SOFT, labelsize=9)
            plt.tight_layout(); st.pyplot(fig_cm); plt.close(fig_cm)

    cmap_svm = LinearSegmentedColormap.from_list('svm_pastel', ['#FFFFFF', '#C4B5FD', PASTEL_LAVENDER])
    cmap_rf  = LinearSegmentedColormap.from_list('rf_pastel',  ['#FFFFFF', PASTEL_BLUE_LIGHT, PASTEL_BLUE])

    render_model_card(col_svm, "SVM",           y_pred_svm, ['#818CF8', PASTEL_LAVENDER], cmap_svm)
    render_model_card(col_rf,  "Random Forest", y_pred_rf,  [PASTEL_GREEN, PASTEL_BLUE],   cmap_rf)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Perbandingan Akurasi</div>", unsafe_allow_html=True)
    fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
    fig_bar.patch.set_facecolor('#F8FBFF'); ax_bar.set_facecolor('#F8FBFF')
    nilai_akurasi = [acc_svm * 100, acc_rf * 100]
    warna_akurasi = [PASTEL_LAVENDER, PASTEL_BLUE]
    for i, (v, c) in enumerate(zip(nilai_akurasi, warna_akurasi)):
        gradient_hbar(ax_bar, y=i, width=v, height=0.45, color_hex=c)
        ax_bar.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=10, fontweight='700', color=INK)
    ax_bar.set_ylim(-0.7, 1.7)
    ax_bar.set_yticks([0, 1]); ax_bar.set_yticklabels(['SVM', 'Random Forest'])
    ax_bar.set_xlim(0, 108)
    ax_bar.spines[['top','right','left']].set_visible(False)
    ax_bar.spines['bottom'].set_color('#E2E8F0')
    ax_bar.xaxis.grid(True, color='#EAF1FB', linewidth=1); ax_bar.set_axisbelow(True)
    ax_bar.tick_params(colors=INK_SOFT)
    plt.tight_layout(); st.pyplot(fig_bar); plt.close(fig_bar)

# PAGE: WORD CLOUD
elif "Word Cloud" in page:

    st.markdown("""
    <div class="main-header">
        <h1>☁️ Word Cloud</h1>
        <p>Kata-kata yang paling sering diasosiasikan dengan setiap kelas sentimen</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Word Cloud per Kelas Sentimen</div>", unsafe_allow_html=True)

    wc_cols   = st.columns(3)
    wc_config = [
        ('positive', '😊 Sentimen Positif', 'BuGn',  'badge-pos'),
        ('neutral',  '😐 Sentimen Netral',  'PuBu',   'badge-neu'),
        ('negative', '😞 Sentimen Negatif', 'RdPu',    'badge-neg'),
    ]
    for (col, (label, title, cmap, badge)) in zip(wc_cols, wc_config):
        with col:
            st.markdown(f"<span class='{badge}'>{title}</span><br><br>", unsafe_allow_html=True)
            text = ' '.join(clean_df[clean_df['polarity'] == label]['text_akhir'].fillna(''))
            if text.strip():
                wc = WordCloud(width=700, height=380, background_color='white',
                               colormap=cmap, max_words=150,
                               prefer_horizontal=0.9, collocations=False).generate(text)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 3.5))
                fig_wc.patch.set_facecolor('white')
                ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off')
                plt.tight_layout(pad=0); st.pyplot(fig_wc); plt.close(fig_wc)
            else:
                st.warning(f"Tidak ada data untuk {label}.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Top 10 Kata Paling Sering Muncul</div>", unsafe_allow_html=True)

    top_cols = st.columns(3)
    for (col, (label, title, _, _)) in zip(top_cols, wc_config):
        with col:
            text  = ' '.join(clean_df[clean_df['polarity'] == label]['text_akhir'].fillna(''))
            words = pd.Series(text.split()).value_counts().head(10)
            fig_t, ax_t = plt.subplots(figsize=(4.5, 4))
            fig_t.patch.set_facecolor('#F8FBFF'); ax_t.set_facecolor('#F8FBFF')
            w_vals = words.values[::-1]
            w_labels = words.index[::-1]
            for i, v in enumerate(w_vals):
                gradient_hbar(ax_t, y=i, width=v, height=0.6, color_hex=COLOR_MAP[label])
            ax_t.set_yticks(range(len(w_labels))); ax_t.set_yticklabels(w_labels)
            ax_t.set_xlim(0, max(w_vals) * 1.15)
            ax_t.set_ylim(-0.6, len(w_labels) - 0.4)
            ax_t.set_title(LABEL_MAP[label], fontsize=10, color=INK)
            ax_t.spines[['top','right','left']].set_visible(False)
            ax_t.spines['bottom'].set_color('#E2E8F0')
            ax_t.xaxis.grid(True, color='#EAF1FB', linewidth=1); ax_t.set_axisbelow(True)
            ax_t.tick_params(colors=INK_SOFT, labelsize=8)
            plt.tight_layout(); st.pyplot(fig_t); plt.close(fig_t)

# PAGE: COBA PREDIKSI
elif "Prediksi" in page:

    st.markdown("""
    <div class="main-header">
        <h1>🔍 Coba Prediksi</h1>
        <p>Masukkan komentar untuk dianalisis sentimennya secara langsung</p>
    </div>""", unsafe_allow_html=True)

    CONTOH = {
        "😊 Contoh Positif": "Setuju banget pembatasan media sosial untuk anak di bawah 16 tahun, itu penting untuk melindungi mereka dari konten berbahaya",
        "😐 Contoh Netral":  "Pemerintah sedang membahas kebijakan pembatasan penggunaan media sosial bagi anak di bawah 16 tahun",
        "😞 Contoh Negatif": "Tidak setuju dengan pembatasan media sosial, anak juga punya hak untuk mengakses informasi dan hiburan",
    }

    col_ex1, col_ex2, col_ex3 = st.columns(3)
    for col, (label, text) in zip([col_ex1, col_ex2, col_ex3], CONTOH.items()):
        if col.button(label, use_container_width=True):
            st.session_state['try_input'] = text

    if 'try_input' not in st.session_state:
        st.session_state['try_input'] = ""

    text_input = st.text_area(
        "Masukkan komentar di sini",
        value=st.session_state['try_input'],
        height=120,
        placeholder="Contoh: Saya setuju dengan pembatasan media sosial untuk anak karena dapat melindungi mereka..."
    )

    if st.button("🔍  Analisis Sentimen", type="primary", use_container_width=True):
        if not text_input.strip():
            st.error("Masukkan komentar terlebih dahulu.")
        else:
            with st.spinner("Menganalisis sentimen..."):
                # Preprocessing sederhana
                clean_text = simple_preprocess(text_input)
                tokens     = clean_text.split()

                # Prediksi lexicon
                lex_score, lex_label = predict_lexicon(tokens, lex_pos, lex_neg)

                # Prediksi ML
                vec     = tfidf.transform([clean_text])
                pred_svm = svm_model.predict(vec)[0]
                pred_rf  = rf_model.predict(vec)[0]

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("### Hasil Analisis")

            c1, c2, c3 = st.columns(3)
            for col, model_name, pred in [
                (c1, "InSet Lexicon",   lex_label),
                (c2, "SVM",             pred_svm),
                (c3, "Random Forest",   pred_rf),
            ]:
                with col:
                    color_map_card = {'positive': PASTEL_GREEN_LIGHT, 'negative': PASTEL_PINK_LIGHT, 'neutral': PASTEL_BLUE_LIGHT}
                    border_map     = {'positive': '#6EE7B7', 'negative': '#FDA4AF', 'neutral': '#93C5FD'}
                    text_map       = {'positive': '#047857', 'negative': '#BE123C', 'neutral': '#1D4ED8'}
                    st.markdown(f"""
                    <div style="background:{color_map_card.get(pred,'#F7FAFF')};
                                border:2px solid {border_map.get(pred,'#E2E8F0')};
                                border-radius:16px;padding:1.5rem;text-align:center;">
                        <div style="font-size:3rem;line-height:1;margin-bottom:0.5rem">{EMOJI_MAP.get(pred,'❓')}</div>
                        <div style="font-size:1.3rem;font-weight:700;color:{text_map.get(pred,INK)}">{LABEL_MAP.get(pred, pred)}</div>
                        <div style="font-size:0.72rem;color:#7C8CA6;text-transform:uppercase;
                                    letter-spacing:0.05em;margin-top:0.3rem">{model_name}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("### Detail Pemrosesan Teks")
            st.write("**Teks asli:**", text_input)
            st.write("**Teks setelah preprocessing:**", clean_text)
            st.write("**Token:**", tokens)
            st.write("**Skor lexicon:**", lex_score)

            # Ringkasan
            preds = [lex_label, pred_svm, pred_rf]
            majority = max(set(preds), key=preds.count)
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.info(f"**Kesimpulan:** Mayoritas model (2 dari 3) memprediksi komentar ini sebagai **{LABEL_MAP.get(majority, majority)}** {EMOJI_MAP.get(majority,'')}")


# FOOTER
st.markdown("""
<div class="footer">
    Analisis Sentimen Publik · SVM &amp; Random Forest · CRISP-DM
</div>
""", unsafe_allow_html=True)