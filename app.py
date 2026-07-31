import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

st.set_page_config(page_title="ACCCIM Neural Engine", page_icon="🧬", layout="wide")

# =====================================================================
# 1. MODEL ARCHITECTURE (Matches ACCCIM Multi-Task PyTorch Model)
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

        # Task 1: Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Task 2: Regression Head
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
# 2. LOAD TRAINED MODEL WEIGHTS
# =====================================================================
@st.cache_resource
def load_model():
    model = ACCCIMMultiTaskModel(input_dim=25)
    state_dict = torch.load("model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Error loading model weights ('model.pth'): {e}")

# =====================================================================
# 3. INTERACTIVE PRESET SAMPLES & SESSION STATE
# =====================================================================
st.title("🧬 ACCCIM Neural Engine")
st.subheader("Diagnostic Report & Clinical Triage Dashboard")

# Define preset vectors for validation
SAMPLES = {
    "Normal": "1.5, 1.2, 1.4, 1.1, 1.6, 1.3, 1.5, 1.2, 1.4, 1.1, 1.3, 1.5, 1.2, 1.4, 1.1, 1.3, 1.5, 1.2, 1.4, 1.1, 1.3, 1.5, 1.2, 1.4, 1.1",
    "LUAD": "18.5, 14.2, 12.0, 16.8, 11.4, 15.0, 13.1, 1.8, 1.5, 1.2, 1.4, 1.1, 1.3, 1.5, 1.2, 1.4, 1.1, 1.3, 1.5, 1.2, 1.4, 1.1, 1.3, 1.5, 1.2",
    "LUSC": "1.2, 1.4, 1.1, 1.3, 1.5, 1.2, 1.4, 1.1, 1.3, 1.5, 1.2, 1.4, 19.1, 17.8, 16.4, 18.2, 12.5, 14.0, 10.5, 1.3, 1.5, 1.2, 1.4, 1.1, 1.3"
}

# Initialize input text in session state if not set
if "gene_input" not in st.session_state:
    st.session_state["gene_input"] = SAMPLES["LUAD"]

st.markdown("### 🧪 Load Test Preset Validation Vectors")
col1, col2, col3 = st.columns(3)

if col1.button("🟢 Load Normal Sample"):
    st.session_state["gene_input"] = SAMPLES["Normal"]

if col2.button("🔴 Load LUAD Sample"):
    st.session_state["gene_input"] = SAMPLES["LUAD"]

if col3.button("🟠 Load LUSC Sample"):
    st.session_state["gene_input"] = SAMPLES["LUSC"]

input_text = st.text_area(
    "Paste 25 Raw Gene Expression Log-Counts (Comma-Separated):",
    value=st.session_state["gene_input"],
    height=100
)

# =====================================================================
# 4. INFERENCE & VISUALIZATION PIPELINE
# =====================================================================
if st.button("Run Model Inference", type="primary"):
    if not model_ready:
        st.error("Model weights are missing or invalid!")
        st.stop()
        
    try:
        # Parse inputs
        clean_values = [float(x.strip()) for x in input_text.replace('\n', ',').replace(' ', ',').split(',') if x.strip()]
        if len(clean_values) < 25:
            clean_values += [1.5] * (25 - len(clean_values))  # Neutral baseline padding
        else:
            clean_values = clean_values[:25]

        # Scaler Standardization (Simulates Colab training distribution)
        raw_arr = np.array(clean_values, dtype=np.float32)
        
        # Scaling strategy matching training baseline (mean ~1.8, std ~1.2 across dataset)
        norm_arr = (raw_arr - 1.8) / 1.2
        tensor_input = torch.tensor(norm_arr, dtype=torch.float32).unsqueeze(0)

        # Forward Pass
        with torch.no_grad():
            logits, reg_out, _ = model(tensor_input)
            probs = F.softmax(logits, dim=1).numpy()[0]
            pathway_score = float(reg_out.numpy()[0][0])
            pred_class_id = int(torch.argmax(logits, dim=1).item())

        class_map = {
            0: "Normal / Benign Baseline", 
            1: "Lung Adenocarcinoma (LUAD)", 
            2: "Lung Squamous Cell (LUSC)"
        }

        # Results Display
        st.success("PyTorch Multi-Task Forward Pass Completed!")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown(f"### **Predicted Histology:**\n#### {class_map[pred_class_id]}")
            st.write(f"**Top Class Confidence:** {probs[pred_class_id]*100:.2f}%")
            st.write(f"**Driver Pathway Load Score:** {pathway_score:.3f}")
            st.progress(pathway_score)

        with res_col2:
            st.markdown("### **Class Probability Breakdown**")
            st.write(f"- **Normal:** {probs[0]*100:.2f}%")
            st.write(f"- **LUAD:** {probs[1]*100:.2f}%")
            st.write(f"- **LUSC:** {probs[2]*100:.2f}%")

        st.divider()

        # Clinical Triage Badge
        if pred_class_id == 0:
            st.info("🟢 **ROUTINE CARE** — Non-Malignant / Baseline Genomic Profile")
        elif pred_class_id == 1:
            st.error("🔴 **CRITICAL URGENCY** — Malignant LUAD Signature (Route to Thoracic Oncology for TKI Evaluation)")
        else:
            st.warning("🟠 **HIGH URGENCY** — Malignant LUSC Signature (Route for Immunotherapy / Pembrolizumab Evaluation)")

    except Exception as e:
        st.error(f"Inference Error: {e}")
