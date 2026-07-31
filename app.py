# =====================================================================
# 1. IMPORTS & CONFIGURATION (MUST BE AT THE TOP)
# =====================================================================
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set Streamlit page configuration FIRST
st.set_page_config(
    page_title="ACCCIM Neural Engine", 
    page_icon="🧬", 
    layout="wide"
)

# =====================================================================
# 2. MODEL ARCHITECTURE (Matches PyTorch Model Weights)
# =====================================================================
class ACCCIMMultiTaskModel(nn.Module):
    def __init__(self, input_dim=25, hidden_dim1=256, hidden_dim2=128):
        super(ACCCIMMultiTaskModel, self).__init__()
        
        # Shared Encoder Backbone
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU()
        )
        
        # Task 1: Classification Head (0: Normal, 1: LUAD, 2: LUSC)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Task 2: Regression Head (Driver Pathway Score [0.0 - 1.0])
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        clf_logits = self.classification_head(embeddings)
        reg_output = self.regression_head(embeddings)
        return clf_logits, reg_output, embeddings

# =====================================================================
# 3. LOAD MODEL WEIGHTS
# =====================================================================
@st.cache_resource
def load_model():
    model = ACCCIMMultiTaskModel(input_dim=25)
    # Load model state weights from the same folder
    state_dict = torch.load("model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_model()
    model_ready = True
except Exception as e:
    model_ready = False

# =====================================================================
# 4. USER INTERFACE & PRESETS
# =====================================================================
st.title("🧬 ACCCIM Neural Engine")
st.subheader("25-Gene Genomic Histology & Driver Pathway Classifier")

if not model_ready:
    st.error("⚠️ `model.pth` file not found or corrupted. Please ensure `model.pth` is present in your GitHub repository root folder.")

# Preset Signatures for quick testing
st.markdown("**Quick Preset Testing:**")
col_p1, col_p2, col_p3 = st.columns(3)

default_input = "8.42, 7.95, 9.18, 6.87, 8.74, 7.53, 9.01, 8.26, 7.68, 8.91, 9.34, 7.12, 8.55, 8.03, 7.81, 9.09, 8.38, 7.47, 8.69, 9.15, 7.74, 8.27, 8.84, 7.96, 8.51"

if "input_val" not in st.session_state:
    st.session_state.input_val = default_input

if col_p1.button("🟢 Normal / Baseline"):
    st.session_state.input_val = default_input

if col_p2.button("🔴 LUAD (KEAP1 Spike)"):
    st.session_state.input_val = "2.1, 1.5, 0.4, 0.8, 1.1, 1.9, 0.5, 0.3, 1.2, 0.4, 18.5, 0.9, 0.1, 0.4, 0.7, 0.3, 0.5, 0.2, 0.1, 0.8, 0.4, 0.3, 0.6, 0.2, 0.1"

if col_p3.button("🟠 LUSC (SOX2 Spike)"):
    st.session_state.input_val = "1.1, 0.9, 0.4, 0.8, 0.5, 0.3, 0.5, 0.3, 0.2, 0.4, 0.2, 0.9, 17.8, 1.4, 0.7, 0.3, 0.5, 0.2, 0.1, 0.8, 0.4, 0.3, 0.6, 0.2, 0.1"

input_text = st.text_area(
    "Paste 25 Raw Gene Expression Values (comma-separated):",
    value=st.session_state.input_val,
    height=100
)

# =====================================================================
# 5. INFERENCE & DUAL-HEAD DISPLAY PIPELINE
# =====================================================================
if st.button("Run Model Inference", type="primary"):
    if not model_ready:
        st.error("Cannot run inference because model weights failed to load!")
        st.stop()
        
    try:
        # 1. Parse raw gene inputs
        clean_values = [float(x.strip()) for x in input_text.replace('\n', ',').replace(' ', ',').split(',') if x.strip()]
        if len(clean_values) < 25:
            clean_values += [1.0] * (25 - len(clean_values))  # Neutral padding
        else:
            clean_values = clean_values[:25]

        # 2. COLAB-MATCHED LOG NORMALIZATION PIPELINE
        raw_arr = np.array(clean_values, dtype=np.float32)
        raw_tensor = torch.tensor(raw_arr, dtype=torch.float32)
        
        # Log2 Transform + Sample-level Z-Score Normalization
        log_tensor = torch.log2(raw_tensor + 1.0)
        std_val = log_tensor.std()
        std_safe = std_val + 1e-6 if std_val != 0 else 1.0
        z_score_tensor = (log_tensor - log_tensor.mean()) / std_safe
        
        model_input = z_score_tensor.unsqueeze(0)

        # 3. PyTorch Forward Pass
        with torch.no_grad():
            logits, reg_out, _ = model(model_input)
            probs = F.softmax(logits, dim=1).numpy()[0]
            pathway_score = float(reg_out.numpy()[0][0])
            pred_class_id = int(torch.argmax(logits, dim=1).item())

        # Subtype Display Map
        class_map = {
            0: "Normal Baseline / Control", 
            1: "Lung Adenocarcinoma (LUAD)", 
            2: "Lung Squamous Cell Carcinoma (LUSC)"
        }

        # Gene Panel Mapping
        luad_genes = ["EGFR", "KRAS", "ALK", "MET", "ROS1", "RET", "ERBB2", "BRAF", "TP53", "STK11", "KEAP1", "NKX2-1"]
        lusc_genes = ["SOX2", "TP63", "KRT5", "KRT6A", "PIK3CA", "FGFR1", "CDKN2A"]

        # Determine Driver Mutation Status based on class and gene spikes
        if pred_class_id == 0 or pathway_score < 0.25:
            driver_head_status = "None Detected"
        elif pred_class_id == 1:
            top_gene_idx = np.argmax(raw_arr[:12])
            driver_head_status = f"{luad_genes[top_gene_idx]} Amplification / Variant Target"
        else:
            top_gene_idx = np.argmax(raw_arr[12:19])
            driver_head_status = f"{lusc_genes[top_gene_idx]} Lineage Driver Amplification"

        # Results Display
        st.success("PyTorch Forward Pass Completed Successfully!")
        st.markdown("### 📊 Dual-Head Neural Engine Report")
        
        st.code(
            f"• Histology Head       : {class_map[pred_class_id]}\n"
            f"• Driver Mutation Head : {driver_head_status}",
            language="text"
        )

        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.write(f"**Top Class Confidence:** `{probs[pred_class_id]*100:.2f}%`")
            st.write(f"**Driver Pathway Load Score:** `{pathway_score:.3f}`")
            st.progress(pathway_score)

        with res_col2:
            st.markdown("**Class Probability Breakdown:**")
            st.write(f"- Normal Baseline: `{probs[0]*100:.2f}%`")
            st.write(f"- LUAD: `{probs[1]*100:.2f}%`")
            st.write(f"- LUSC: `{probs[2]*100:.2f}%`")

        st.divider()

        # Clinical Triage Badge
        if pred_class_id == 0:
            st.info("🟢 **ROUTINE CARE** — Non-Malignant / Baseline Genomic Profile")
        elif pred_class_id == 1:
            st.error("🔴 **CRITICAL URGENCY** — Malignant LUAD Signature (Route to Thoracic Oncology for Targeted TKI Evaluation)")
        else:
            st.warning("🟠 **HIGH URGENCY** — Malignant LUSC Signature (Route for Immunotherapy Evaluation)")

    except Exception as e:
        st.error(f"Inference Error: {e}")
