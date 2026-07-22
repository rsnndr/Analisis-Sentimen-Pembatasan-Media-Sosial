import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Media Sosial",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header utama */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}
.main-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    line-height: 1.3;
    color: white;
}
.main-header p {
    font-size: 0.9rem;
    color: #a0aec0;
    margin: 0;
}

/* Metric card */
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}
.metric-card .metric-sub {
    font-size: 0.75rem;
    color: #a0aec0;
    margin-top: 0.2rem;
}

/* Section header */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

/* Badge sentimen */
.badge-pos {
    background: #f0fff4;
    color: #276749;
    border: 1px solid #9ae6b4;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-neg {
    background: #fff5f5;
    color: #9b2c2c;
    border: 1px solid #feb2b2;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f7fafc;
    padding: 6px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Model card */
.model-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
}
.model-card h4 {
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
    color: #2d3748;
}

/* Accuracy highlight */
.acc-highlight {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    margin-bottom: 1rem;
}
.acc-highlight .acc-val {
    font-size: 1.8rem;
    font-weight: 700;
}
.acc-highlight .acc-label {
    font-size: 0.75rem;
    opacity: 0.85;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #a0aec0;
    font-size: 0.78rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("hasil_preprocessing.csv")
    return df

# ── TRAIN MODEL ─────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    X = df['text_akhir'].fillna('')
    y = df['polarity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)
    y_pred_rf = rf_model.predict(X_test_tfidf)

    return y_test, y_pred_svm, y_pred_rf, svm_model, rf_model, X_test_tfidf

# ── LOAD ─────────────────────────────────────────────────────
with st.spinner("Memuat data dan melatih model..."):
    try:
        clean_df = load_data()
        y_test, y_pred_svm, y_pred_rf, svm_model, rf_model, X_test_tfidf = train_models(clean_df)
        data_loaded = True
    except Exception as e:
        data_loaded = False
        st.error(f"Gagal memuat data: {e}. Pastikan file `hasil_preprocessing.csv` ada di folder yang sama.")
        st.stop()

# ── HEADER ───────────────────────────────────────────────────
sentimen_counts = clean_df['polarity'].value_counts()
total_data      = len(clean_df)
jml_pos         = sentimen_counts.get('positive', 0)
jml_neg         = sentimen_counts.get('negative', 0)
acc_svm         = accuracy_score(y_test, y_pred_svm)
acc_rf          = accuracy_score(y_test, y_pred_rf)

st.markdown("""
<div class="main-header">
    <h1>💬 Analisis Sentimen Publik</h1>
    <p>Pembatasan Media Sosial bagi Anak di Bawah 16 Tahun </p>
</div>
""", unsafe_allow_html=True)

# ── METRIC SUMMARY ───────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#2d3748">{total_data:,}</div>
        <div class="metric-label">Total Komentar</div>
        <div class="metric-sub">3 platform</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#276749">{jml_pos:,}</div>
        <div class="metric-label">Sentimen Positif</div>
        <div class="metric-sub">{jml_pos/total_data*100:.1f}% dari total</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#9b2c2c">{jml_neg:,}</div>
        <div class="metric-label">Sentimen Negatif</div>
        <div class="metric-sub">{jml_neg/total_data*100:.1f}% dari total</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#553c9a">{acc_svm*100:.1f}%</div>
        <div class="metric-label">Akurasi SVM</div>
        <div class="metric-sub">kernel linear</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#2b6cb0">{acc_rf*100:.1f}%</div>
        <div class="metric-label">Akurasi RF</div>
        <div class="metric-sub">100 pohon</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Distribusi Sentimen", "🤖  Evaluasi Model", "☁️  Word Cloud"])

# ───────── TAB 1: DISTRIBUSI ─────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>Distribusi Sentimen Keseluruhan</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#f8fafc')

        colors  = ['#48bb78', '#fc8181']
        bars    = ax.bar(sentimen_counts.index, sentimen_counts.values,
                         color=colors, width=0.5, edgecolor='white', linewidth=1.5)

        for bar, val in zip(bars, sentimen_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                    f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='600',
                    color='#2d3748')

        ax.set_xlabel('Sentimen', fontsize=10, color='#4a5568')
        ax.set_ylabel('Jumlah Komentar', fontsize=10, color='#4a5568')
        ax.tick_params(colors='#4a5568')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#e2e8f0')
        ax.yaxis.grid(True, color='#e2e8f0', linewidth=0.8)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#f8fafc')
        ax2.set_facecolor('#f8fafc')

        wedges, texts, autotexts = ax2.pie(
            sentimen_counts.values,
            labels=sentimen_counts.index,
            autopct='%1.1f%%',
            colors=['#48bb78', '#fc8181'],
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2),
            pctdistance=0.75
        )
        for t in texts:
            t.set_fontsize(11)
            t.set_color('#2d3748')
            t.set_fontweight('600')
        for a in autotexts:
            a.set_fontsize(10)
            a.set_color('white')
            a.set_fontweight('700')
        plt.tight_layout()
        st.pyplot(fig2)


# ───────── TAB 2: EVALUASI MODEL ─────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Perbandingan Performa Model</div>", unsafe_allow_html=True)

    col_svm, col_rf = st.columns(2)

    def render_model_card(col, nama, y_pred, model, label_color):
        with col:
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            st.markdown(f"""
            <div class="acc-highlight" style="background: linear-gradient(135deg, {label_color[0]}, {label_color[1]});">
                <div class="acc-val">{acc*100:.2f}%</div>
                <div class="acc-label">Akurasi {nama}</div>
            </div>""", unsafe_allow_html=True)

            # Metrik tabel ringkas
            m1, m2, m3 = st.columns(3)
            for col_m, kelas in zip([m1, m2, m3], ['negative', 'positive', 'macro avg']):
                if kelas in report:
                    label_display = kelas.replace(' avg', '').title()
                    with col_m:
                        st.markdown(f"""
                        <div style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;
                                    padding:0.6rem;text-align:center;margin-bottom:0.5rem">
                            <div style="font-size:0.7rem;color:#718096;text-transform:uppercase;
                                        letter-spacing:0.04em;font-weight:500">{label_display}</div>
                            <div style="font-size:1rem;font-weight:700;color:#2d3748">
                                {report[kelas]['f1-score']:.2f}</div>
                            <div style="font-size:0.65rem;color:#a0aec0">F1-score</div>
                        </div>""", unsafe_allow_html=True)

            # Confusion matrix
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            fig_cm.patch.set_facecolor('#f8fafc')
            disp = ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, ax=ax_cm,
                cmap='Blues', colorbar=False
            )
            ax_cm.set_title(f'Confusion Matrix — {nama}', fontsize=10, color='#2d3748', pad=10)
            ax_cm.tick_params(colors='#4a5568', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig_cm)

    render_model_card(col_svm, "SVM",           y_pred_svm, svm_model, ['#667eea', '#764ba2'])
    render_model_card(col_rf,  "Random Forest", y_pred_rf,  rf_model,  ['#2193b0', '#6dd5ed'])

    # Perbandingan akurasi bar chart
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Perbandingan Akurasi</div>", unsafe_allow_html=True)

    fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
    fig_bar.patch.set_facecolor('#f8fafc')
    ax_bar.set_facecolor('#f8fafc')
    models_name = ['SVM', 'Random Forest']
    accs        = [acc_svm * 100, acc_rf * 100]
    bars_cmp    = ax_bar.barh(models_name, accs, color=['#7f8ef5', '#63b3ed'],
                               edgecolor='white', height=0.4)
    for bar, val in zip(bars_cmp, accs):
        ax_bar.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontsize=10, fontweight='600', color='#2d3748')
    ax_bar.set_xlim(0, 105)
    ax_bar.spines[['top', 'right', 'left']].set_visible(False)
    ax_bar.spines['bottom'].set_color('#e2e8f0')
    ax_bar.xaxis.grid(True, color='#e2e8f0', linewidth=0.8)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(colors='#4a5568')
    plt.tight_layout()
    st.pyplot(fig_bar)


# ───────── TAB 3: WORD CLOUD ──────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Word Cloud per Kelas Sentimen</div>", unsafe_allow_html=True)

    col_wc1, col_wc2 = st.columns(2)

    def render_wordcloud(col, label, title, colormap, badge_class):
        with col:
            st.markdown(f"<span class='{badge_class}'>{title}</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            text = ' '.join(clean_df[clean_df['polarity'] == label]['text_akhir'].fillna(''))
            if text.strip():
                wc = WordCloud(
                    width=800, height=420,
                    background_color='white',
                    colormap=colormap,
                    max_words=150,
                    prefer_horizontal=0.9,
                    collocations=False
                ).generate(text)
                fig_wc, ax_wc = plt.subplots(figsize=(7, 3.8))
                fig_wc.patch.set_facecolor('white')
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig_wc)
            else:
                st.warning(f"Tidak ada data untuk sentimen {label}.")

    render_wordcloud(col_wc1, 'positive', '😊 Sentimen Positif', 'Greens',  'badge-pos')
    render_wordcloud(col_wc2, 'negative', '😞 Sentimen Negatif', 'Reds',    'badge-neg')

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Top 10 kata
    st.markdown("<div class='section-header'>Top 10 Kata Paling Sering Muncul</div>", unsafe_allow_html=True)
    col_top1, col_top2 = st.columns(2)

    def render_top_words(col, label, color):
        with col:
            text = ' '.join(clean_df[clean_df['polarity'] == label]['text_akhir'].fillna(''))
            words = pd.Series(text.split()).value_counts().head(10)
            fig_t, ax_t = plt.subplots(figsize=(5, 4))
            fig_t.patch.set_facecolor('#f8fafc')
            ax_t.set_facecolor('#f8fafc')
            ax_t.barh(words.index[::-1], words.values[::-1], color=color,
                      edgecolor='white', linewidth=1)
            ax_t.spines[['top', 'right', 'left']].set_visible(False)
            ax_t.spines['bottom'].set_color('#e2e8f0')
            ax_t.xaxis.grid(True, color='#e2e8f0', linewidth=0.8)
            ax_t.set_axisbelow(True)
            ax_t.tick_params(colors='#4a5568', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig_t)

    render_top_words(col_top1, 'positive', '#48bb78')
    render_top_words(col_top2, 'negative', '#fc8181')