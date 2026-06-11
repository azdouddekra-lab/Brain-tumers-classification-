import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background-color: #080c14; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #0a2540 100%);
    padding: 2.2rem 2.8rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid #1e3a5f;
    box-shadow: 0 4px 32px rgba(59,130,246,0.08);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header h1 {
    color: #f0f6ff;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #7fa3c8;
    margin: 0.5rem 0 0 0;
    font-size: 0.92rem;
    font-weight: 400;
}

/* ── Cards ── */
.card {
    background: #0f1623;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
}
.card h3 {
    color: #e2e8f0;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 1.1rem 0;
    letter-spacing: 0.3px;
}

/* ── Result boxes ── */
.result-glioma     { background: linear-gradient(135deg,#2d0a0a,#3d0f0f); border: 2px solid #ef4444; color: #fca5a5; }
.result-meningioma { background: linear-gradient(135deg,#2a2000,#352900); border: 2px solid #f59e0b; color: #fde68a; }
.result-pituitary  { background: linear-gradient(135deg,#0a1a35,#0d2040); border: 2px solid #3b82f6; color: #93c5fd; }
.result-no_tumor   { background: linear-gradient(135deg,#062010,#082918); border: 2px solid #22c55e; color: #86efac; }

.result-box {
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 1rem 0;
    letter-spacing: 0.5px;
}

/* ── Confidence bars ── */
.conf-bar-bg {
    background: #1a2540;
    border-radius: 999px;
    height: 8px;
    margin: 4px 0 10px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* ── Metrics ── */
.metric-row { display: flex; gap: 0.8rem; margin-top: 0.8rem; }
.metric-item {
    flex: 1;
    background: #131c2e;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 0.8rem 0.6rem;
    text-align: center;
}
.metric-val { font-size: 1.3rem; font-weight: 700; color: #60a5fa; }
.metric-lbl { font-size: 0.72rem; color: #4a6080; margin-top: 2px; letter-spacing: 0.3px; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    letter-spacing: 0.4px;
}
.badge-blue   { background: #0e2a4a; color: #60a5fa; border: 1px solid #1e4a8a; }
.badge-green  { background: #052e16; color: #4ade80; border: 1px solid #15803d; }
.badge-purple { background: #1e0a40; color: #c084fc; border: 1px solid #6d28d9; }
.badge-gold   { background: #2a1a00; color: #fbbf24; border: 1px solid #b45309; }

/* ── Best model banner ── */
.best-model-banner {
    background: linear-gradient(90deg, #052e16, #0a3d1f);
    border: 1px solid #16a34a;
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.82rem;
    color: #86efac;
}

/* ── Eval section ── */
.eval-card {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.eval-model-name { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem; }
.eval-bar-bg {
    background: #131c2e;
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
    margin: 3px 0;
}
.eval-bar-fill {
    height: 12px;
    border-radius: 999px;
}

/* ── Grad-CAM section ── */
.gcam-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #93c5fd;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #0f1623;
    border: 2px dashed #1e3a5f;
    border-radius: 14px;
    padding: 0.8rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080c14;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #7fa3c8;
    font-size: 0.83rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1623;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #93c5fd !important;
}

/* ── Disclaimer ── */
.disclaimer {
    margin-top: 1.8rem;
    padding: 0.9rem 1.4rem;
    background: #0a1628;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    font-size: 0.79rem;
    color: #4a6080;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

CLASS_INFO = {
    'glioma': {
        'emoji': '🔴', 'label': 'Gliome', 'color': '#ef4444', 'css': 'result-glioma',
        'desc': 'Tumeur maligne des cellules gliales. Représente ~30% des tumeurs cérébrales.',
        'severity': 'Élevée', 'sev_color': '#ef4444'
    },
    'meningioma': {
        'emoji': '🟡', 'label': 'Méningiome', 'color': '#f59e0b', 'css': 'result-meningioma',
        'desc': 'Tumeur des méninges, souvent bénigne et à croissance lente.',
        'severity': 'Modérée', 'sev_color': '#f59e0b'
    },
    'pituitary': {
        'emoji': '🔵', 'label': 'Tumeur hypophysaire', 'color': '#3b82f6', 'css': 'result-pituitary',
        'desc': "Tumeur de la glande pituitaire, généralement bénigne et traitable.",
        'severity': 'Faible', 'sev_color': '#3b82f6'
    },
    'no_tumor': {
        'emoji': '🟢', 'label': 'Aucune tumeur', 'color': '#22c55e', 'css': 'result-no_tumor',
        'desc': "Aucune anomalie tumorale détectée sur l'IRM analysée.",
        'severity': 'Aucune', 'sev_color': '#22c55e'
    }
}

DRIVE_IDS = {
    'MobileNetV2 Fine Tuning ⭐ Meilleur': '1uGofq96oRk6E5_9vYhVqejarUijrpQMc',
    'MobileNetV2 Transfer Learning':        '1dHhJq3x5yyCMSJThv7mVF5OBd4Nqjruc',
    'VGG16 Fine Tuning':                    '1AZQTwYCb3w3p3YJzKW_WWl91UmmzKgeU',
    'VGG16 Transfer Learning':              '1CNa4-I_sVUdlj4npBcPXNbWrvHwmT-VI',
    'CNN Custom (Baseline)':                '1jgNpnvfvQMyyjjLI5mrgyhrdN7E4e8If',
}

MODEL_INFO = {
    'MobileNetV2 Fine Tuning ⭐ Meilleur': {
        'file': 'mobilenet_ft.h5', 'acc': 96.21, 'f1': 0.96,
        'badge': 'badge-green', 'badge_txt': '⭐ Meilleur modèle',
        'params': '2.4M', 'size': '14 MB',
        'desc': 'Fine Tuning sur 30 dernières couches — léger, précis, déployable',
        'last_layer': 'Conv_1'
    },
    'MobileNetV2 Transfer Learning': {
        'file': 'mobilenet_tl.h5', 'acc': 87.67, 'f1': 0.89,
        'badge': 'badge-blue', 'badge_txt': 'Transfer Learning',
        'params': '2.4M', 'size': '14 MB',
        'desc': 'MobileNetV2 pré-entraîné — base gelée, tête entraînée',
        'last_layer': 'Conv_1'
    },
    'VGG16 Fine Tuning': {
        'file': 'vgg16_ft.h5', 'acc': 96.13, 'f1': 0.96,
        'badge': 'badge-blue', 'badge_txt': 'Fine Tuning',
        'params': '21M', 'size': '82 MB',
        'desc': 'VGG16 fine-tunée — robuste, haute précision',
        'last_layer': 'block5_conv3'
    },
    'VGG16 Transfer Learning': {
        'file': 'vgg16_tl.h5', 'acc': 88.42, 'f1': 0.89,
        'badge': 'badge-purple', 'badge_txt': 'Transfer Learning',
        'params': '21M', 'size': '82 MB',
        'desc': 'VGG16 pré-entraîné — base gelée',
        'last_layer': 'block5_conv3'
    },
    'CNN Custom (Baseline)': {
        'file': 'cnn_custom.h5', 'acc': 84.79, 'f1': 0.83,
        'badge': 'badge-purple', 'badge_txt': 'Baseline',
        'params': '26M', 'size': '99 MB',
        'desc': 'CNN 3 blocs Conv2D + BatchNorm — entraîné from scratch',
        'last_layer': 'conv2d_5'
    },
}

# Evaluation data (real results from PDF)
EVAL_DATA = {
    'MobileNetV2 FT':  {'acc': 96.21, 'color': '#4ade80'},
    'VGG16 FT':        {'acc': 96.13, 'color': '#60a5fa'},
    'VGG16 TL':        {'acc': 88.42, 'color': '#a78bfa'},
    'MobileNetV2 TL':  {'acc': 87.67, 'color': '#f59e0b'},
    'CNN Custom':      {'acc': 84.79, 'color': '#94a3b8'},
}

# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    try:
        import tensorflow as tf
        model_file = MODEL_INFO[model_name]['file']
        if not os.path.exists(model_file):
            drive_id = DRIVE_IDS[model_name]
            url = f"https://drive.google.com/uc?id={drive_id}"
            with st.spinner("⏳ Téléchargement du modèle depuis Google Drive..."):
                gdown.download(url, model_file, quiet=False)
        model = tf.keras.models.load_model(model_file)
        return model
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle : {e}")
        return None

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB').resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def predict(model, img_array):
    preds = model.predict(img_array, verbose=0)[0]
    return CLASSES[int(np.argmax(preds))], preds

# ─── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array, pred_class_idx, last_conv_layer_name):
    try:
        import tensorflow as tf
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            loss = predictions[:, pred_class_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception as e:
        return None

def apply_gradcam_overlay(original_img: Image.Image, heatmap: np.ndarray) -> Image.Image:
    import cv2
    img_np = np.array(original_img.convert('RGB').resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.55, heatmap_colored, 0.45, 0)
    return Image.fromarray(overlay)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Brain Tumor Detection")
    st.markdown("---")
    st.markdown("**📚 Projet :** PFE Deep Learning")
    st.markdown("**🗂 Dataset :** Brain Tumor MRI (Kaggle)")
    st.markdown("**🖼 Images :** 16 000 (4 × 4 000)")
    st.markdown("**⚙️ Framework :** TensorFlow / Keras")
    st.markdown("**💻 Entraînement :** Google Colab T4")
    st.markdown("---")
    st.markdown("**🏷 Classes :**")
    for cls, info in CLASS_INFO.items():
        st.markdown(
            f"{info['emoji']} **{info['label']}**  \n"
            f"<small style='color:#4a6080'>{info['desc']}</small>",
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem; color:#2d4a6a; text-align:center'>"
        "⭐ Meilleur modèle : MobileNetV2 FT<br>"
        "<span style='color:#4ade80; font-weight:700'>96.21% accuracy</span>"
        "</div>", unsafe_allow_html=True
    )

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧠 Détection des Tumeurs Cérébrales</h1>
    <p>PFE · Deep Learning · CNN + Transfer Learning + Fine Tuning &nbsp;│&nbsp;
       Classification IRM en 4 classes &nbsp;│&nbsp;
       <span style="color:#4ade80; font-weight:600">Meilleur modèle : MobileNetV2 FT — 96.21%</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Diagnostic", "📊 Évaluation des modèles", "ℹ️ À propos"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Best model banner
        st.markdown("""
        <div class="best-model-banner">
            ⭐ <strong>Recommandé :</strong> MobileNetV2 Fine Tuning — 96.21% accuracy
        </div>
        """, unsafe_allow_html=True)

        # Model selector
        st.markdown('<div class="card"><h3>⚙️ Choix du modèle</h3>', unsafe_allow_html=True)
        selected_model = st.selectbox("Modèle", list(MODEL_INFO.keys()), label_visibility="collapsed")
        minfo = MODEL_INFO[selected_model]
        st.markdown(f"""
            <div style="margin-bottom:0.6rem">
                <span class="badge {minfo['badge']}">{minfo['badge_txt']}</span>
                <small style="color:#64748b">{minfo['desc']}</small>
            </div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-val">{minfo['acc']}%</div>
                    <div class="metric-lbl">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="metric-val">{minfo['params']}</div>
                    <div class="metric-lbl">Paramètres</div>
                </div>
                <div class="metric-item">
                    <div class="metric-val">{minfo['size']}</div>
                    <div class="metric-lbl">Taille</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Upload
        st.markdown('<div class="card"><h3>📂 Image IRM</h3>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Glisse ou clique pour uploader une IRM",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="IRM uploadée", use_column_width=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:2rem; color:#1e3a5f;">
                <div style="font-size:3rem;">🩻</div>
                <div style="font-size:0.88rem; margin-top:0.5rem; color:#4a6080">Aucune image sélectionnée</div>
                <div style="font-size:0.75rem; color:#2d4a6a">Formats acceptés : JPG, PNG</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card"><h3>📊 Résultat de l\'analyse</h3>', unsafe_allow_html=True)

        if uploaded:
            with st.spinner("🔬 Analyse en cours..."):
                model = load_model(selected_model)

            if model is not None:
                img_array = preprocess_image(img)
                pred_class, probas = predict(model, img_array)
                info = CLASS_INFO[pred_class]
                pred_idx = CLASSES.index(pred_class)

                # Result badge
                st.markdown(f"""
                <div class="result-box {info['css']}">
                    {info['emoji']}  {info['label'].upper()}
                </div>
                <p style='color:#64748b; font-size:0.82rem; text-align:center; margin-top:-0.3rem'>
                    {info['desc']}
                </p>
                """, unsafe_allow_html=True)

                # Severity
                st.markdown(f"""
                <div style="display:flex; gap:0.6rem; margin-bottom:1rem; justify-content:center">
                    <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
                                padding:0.4rem 0.9rem; font-size:0.78rem; color:#64748b">
                        Sévérité : <span style="color:{info['sev_color']}; font-weight:600">{info['severity']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bars
                st.markdown("**Confiance par classe :**")
                for i, cls in enumerate(CLASSES):
                    pct = float(probas[i]) * 100
                    c_info = CLASS_INFO[cls]
                    is_top = (cls == pred_class)
                    lbl_style = "font-weight:700; color:#e2e8f0;" if is_top else "color:#4a6080;"
                    st.markdown(f"""
                    <div style="margin-bottom:5px">
                        <div style="display:flex; justify-content:space-between; {lbl_style} font-size:0.82rem;">
                            <span>{c_info['emoji']} {c_info['label']}</span>
                            <span>{pct:.1f}%</span>
                        </div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{pct:.1f}%; background:{c_info['color']};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Global confidence
                top_conf = float(np.max(probas)) * 100
                if top_conf >= 90: level, lc = "Très élevée ✅", "#22c55e"
                elif top_conf >= 70: level, lc = "Élevée 🟡", "#f59e0b"
                else: level, lc = "Faible ⚠️", "#ef4444"
                st.markdown(f"""
                <div style="margin-top:0.8rem; padding:0.7rem 1rem; background:#0a1628;
                     border:1px solid #1e3a5f; border-radius:10px;">
                    <span style="color:#4a6080; font-size:0.79rem">Confiance globale : </span>
                    <span style="color:{lc}; font-weight:700">{level} ({top_conf:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)

                # ── Grad-CAM ──
                st.markdown("---")
                st.markdown('<div class="gcam-title">🌡️ Grad-CAM — Zone activée par le modèle</div>', unsafe_allow_html=True)

                last_layer = minfo.get('last_layer', 'conv2d_5')
                heatmap = compute_gradcam(model, img_array, pred_idx, last_layer)

                if heatmap is not None:
                    try:
                        overlay = apply_gradcam_overlay(img, heatmap)
                        gcol1, gcol2 = st.columns(2)
                        with gcol1:
                            st.image(img.resize((224,224)), caption="IRM originale", use_column_width=True)
                        with gcol2:
                            st.image(overlay, caption="🌡️ Grad-CAM", use_column_width=True)
                        st.markdown("""
                        <div style="font-size:0.75rem; color:#2d4a6a; text-align:center; margin-top:0.3rem">
                            🔴 Rouge = zone fortement activée &nbsp;|&nbsp; 🔵 Bleu = zone peu activée
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.info(f"Grad-CAM non disponible : {e}")
                else:
                    st.info("Grad-CAM non disponible pour ce modèle.")

        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 1rem; color:#1e3a5f;">
                <div style="font-size:3.5rem;">🔬</div>
                <div style="font-size:0.95rem; margin-top:1rem; color:#2d4a6a">
                    Uploadez une image IRM<br>pour démarrer l'analyse
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Comparaison des 5 modèles")
    st.markdown("""
    <div style="font-size:0.85rem; color:#4a6080; margin-bottom:1.5rem">
        Tous les modèles ont été entraînés sur le même dataset (16 000 images, 4 classes),
        avec le même split 70/15/15 et évalués sur le même jeu de test (2 400 images).
    </div>
    """, unsafe_allow_html=True)

    # Accuracy bars
    for model_name, data in EVAL_DATA.items():
        is_best = model_name == 'MobileNetV2 FT'
        border = f"border: 1px solid {data['color']};" if is_best else ""
        star = " ⭐" if is_best else ""
        st.markdown(f"""
        <div class="eval-card" style="{border}">
            <div class="eval-model-name">{model_name}{star}</div>
            <div style="display:flex; align-items:center; gap:0.8rem">
                <div class="eval-bar-bg" style="flex:1">
                    <div class="eval-bar-fill"
                         style="width:{data['acc']}%; background:{data['color']}">
                    </div>
                </div>
                <div style="font-size:1rem; font-weight:700; color:{data['color']}; min-width:52px">
                    {data['acc']}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Detailed table
    st.markdown("### 📋 Tableau comparatif détaillé")
    table_data = {
        'Modèle':     ['CNN Custom', 'MobileNetV2 TL', 'VGG16 TL', 'VGG16 FT', 'MobileNetV2 FT ⭐'],
        'Accuracy':   ['84.79%', '87.67%', '88.42%', '96.13%', '96.21%'],
        'Precision':  ['0.84', '0.89', '0.90', '0.96', '0.96'],
        'Recall':     ['0.83', '0.89', '0.89', '0.96', '0.96'],
        'F1-score':   ['0.83', '0.89', '0.89', '0.96', '0.96'],
        'Params':     ['26M', '2.4M', '21M', '21M', '2.4M'],
    }
    import pandas as pd
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="margin-top:1rem; padding:1rem 1.2rem; background:#052e16;
         border:1px solid #16a34a; border-radius:10px; font-size:0.85rem; color:#86efac">
        <strong>🏆 Conclusion :</strong> MobileNetV2 Fine Tuning est le meilleur modèle avec
        <strong>96.21% d'accuracy</strong> et seulement <strong>2.4M paramètres</strong> —
        offrant le meilleur compromis précision / légèreté pour un déploiement réel.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ À propos du projet")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🎯 Objectif</h3>
            <p style="color:#7fa3c8; font-size:0.85rem; line-height:1.7">
            Développer un système de classification automatique des tumeurs
            cérébrales à partir d'images IRM par Deep Learning,
            en comparant 5 architectures différentes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <h3>🗂 Dataset</h3>
            <p style="color:#7fa3c8; font-size:0.85rem; line-height:1.7">
            <strong style="color:#e2e8f0">16 000 images IRM</strong> — 4 classes équilibrées (4 000/classe)<br>
            🔴 Gliome · 🟡 Méningiome · 🔵 Pituitary · 🟢 No Tumor<br>
            Split : <strong style="color:#e2e8f0">70% train / 15% val / 15% test</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>⚙️ Environnement technique</h3>
            <p style="color:#7fa3c8; font-size:0.85rem; line-height:1.7">
            🐍 Python 3.x<br>
            🧠 TensorFlow / Keras<br>
            💻 Google Colab (GPU T4)<br>
            🚀 Streamlit (déploiement)<br>
            📊 NumPy · Pandas · OpenCV
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <h3>📐 Paramètres d'entraînement</h3>
            <p style="color:#7fa3c8; font-size:0.85rem; line-height:1.7">
            Optimizer : <strong style="color:#e2e8f0">Adam</strong><br>
            Loss : <strong style="color:#e2e8f0">Categorical Crossentropy</strong><br>
            Callbacks : EarlyStopping · ReduceLROnPlateau · ModelCheckpoint<br>
            Input size : <strong style="color:#e2e8f0">224 × 224 px</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong style="color:#7fa3c8">Avertissement médical :</strong>
    Cette application est développée à des fins académiques (PFE).
    Les résultats ne remplacent pas un diagnostic médical professionnel.
    Consultez toujours un médecin qualifié pour toute décision médicale.
</div>
""", unsafe_allow_html=True)