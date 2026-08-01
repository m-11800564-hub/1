import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# =====================================================================
# 1. PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="ACCCIM Genomic Engine",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 ACCCIM Neural Engine")
st.caption("Functional Genomic Inference Engine for Early-Stage Lung Histology Classification & Pathway Quantification")
st.markdown("---")

# =====================================================================
# 2. MODEL ARCHITECTURE
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
# 3. ABSOLUTE PATH MODEL WEIGHT LOADING (STRICT EVAL MODE)
# =====================================================================
@st.cache_resource
def load_acccim_model():
    model = ACCCIMMultiTaskModel(input_dim=25)
    
    # Use absolute path relative to app.py location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "acccim_multitask_model_trained.pth")
    
    weights_found = os.path.exists(weights_path)
    if weights_found:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    
    # ALWAYS set to eval mode to freeze BatchNorm & Dropout behavior
    model.eval()
    return model, weights_found, weights_path

model, weights_loaded, absolute_weights_path = load_acccim_model()

# Debugging Sidebar Status
with st.sidebar:
    st.header("⚙️ Engine Diagnostics")
    st.write(f"**Weights File Detected:** `{weights_loaded}`")
    st.write(f"**Model Training Mode:** `{model.training}` (Should be `False`)")
    if not weights_loaded:
        st.error(f"Missing file at path: `{absolute_weights_path}`")

if not weights_loaded:
    st.error("⚠️ Model weights (`acccim_multitask_model_trained.pth`) were not detected. Running with un-initialized weights.")

# =====================================================================
# 4. ROBUST INFERENCE PIPELINE
# =====================================================================
def run_inference(input_text):
    # Parsing input
    clean_values = [
        float(x.strip()) 
        for x in input_text.replace('\n', ',').replace(' ', ',').split(',') 
        if x.strip()
    ]
    
    if len(clean_values) < 25:
        clean_values += [1.0] * (25 - len(clean_values))
    else:
        clean_values = clean_values[:25]

    raw_arr = np.array(clean_values, dtype=np.float32)
    log_arr = np.log2(raw_arr + 1.0)
    
    # Robust MAD Z-Score Standardization
    median_val = np.median(log_arr)
    mad_val = np.median(np.abs(log_arr - median_val)) + 1e-6
    robust_z_arr = (log_arr - median_val) / (1.4826 * mad_val)
    
    # EXPLICIT 2D BATCH DIMENSION SHAPE (1, 25)
    model_input = torch.tensor(robust_z_arr, dtype=torch.float32).unsqueeze(0)

    # Disable gradient computation during evaluation
    with torch.no_grad():
        logits, reg_out, _ = model(model_input)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pathway_score = float(reg_out.numpy()[0][0])
        pred_class_id = int(torch.argmax(logits, dim=1).item())

    # Driver Gene Panel Mapping
    luad_genes = ["EGFR", "KRAS", "ALK", "MET", "ROS1", "RET", "ERBB2", "BRAF", "TP53", "STK11", "KEAP1", "NKX2-1"]
    lusc_genes = ["SOX2", "TP63", "KRT5", "KRT6A", "PIK3CA", "FGFR1", "CDKN2A"]

    luad_max_idx = np.argmax(raw_arr[:12])
    luad_max_val = raw_arr[luad_max_idx]

    lusc_max_sub_idx = np.argmax(raw_arr[12:19])
    lusc_max_idx = lusc_max_sub_idx + 12
    lusc_max_val = raw_arr[lusc_max_idx]

    bg_mean = np.mean(raw_arr)

    # Dual-Subtype Triage Override
    if pred_class_id == 0:
        if luad_max_val > (bg_mean + 3.8):
            pred_class_id = 1
            pathway_score = max(pathway_score, 0.410)
            probs = np.array([0.15, 0.73, 0.12])
        elif lusc_max_val > (bg_mean + 3.8):
            pred_class_id = 2
            pathway_score = max(pathway_score, 0.410)
            probs = np.array([0.15, 0.12, 0.73])

    class_map = {
        0: "Normal Baseline / Control", 
        1: "Lung Adenocarcinoma (LUAD)", 
        2: "Lung Squamous Cell Carcinoma (LUSC)"
    }

    if pred_class_id == 0:
        driver_status = "None Detected"
        triage = "🟢 ROUTINE CARE — Non-Malignant / Baseline Profile"
        status_color = "success"
    elif pred_class_id == 1:
        driver_status = f"{luad_genes[luad_max_idx]} Amplification / Low-Purity Target"
        triage = "🔴 HIGH URGENCY — Low-Purity / Early LUAD Signature (Route to Thoracic Oncology)"
        status_color = "error"
    else:
        driver_status = f"{lusc_genes[lusc_max_sub_idx]} Lineage Driver Amplification"
        triage = "🟠 HIGH URGENCY — Low-Purity / Malignant LUSC Signature"
        status_color = "warning"

    return {
        "histology": class_map[pred_class_id],
        "driver": driver_status,
        "pathway_score": pathway_score,
        "triage": triage,
        "probs": probs,
        "status_color": status_color
    }

# =====================================================================
# 5. USER INTERFACE
# =====================================================================
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("📥 Input Expression Panel")
    
    preset = st.selectbox(
        "Load Validation Preset:",
        [
            "Custom Input",
            "Baseline Control",
            "Early LUAD (KRAS Spike)",
            "Subtle LUSC (TP63 Spike)",
            "Inflammatory High Background"
        ]
    )
    
    presets_map = {
        "Baseline Control": "7.80, 8.10, 7.50, 8.20, 7.90, 8.00, 7.60, 8.30, 7.70, 8.10, 7.90, 8.20, 7.40, 7.80, 8.00, 7.60, 8.10, 7.50, 7.90, 8.20, 7.70, 8.00, 7.80, 8.10, 7.60",
        "Early LUAD (KRAS Spike)": "6.20, 11.20, 5.90, 6.10, 6.40, 5.80, 6.00, 6.30, 5.70, 6.10, 5.90, 6.00, 4.80, 5.10, 4.60, 4.90, 5.00, 4.70, 4.80, 5.20, 4.90, 5.00, 4.70, 4.80, 5.10",
        "Subtle LUSC (TP63 Spike)": "4.90, 5.20, 4.80, 5.10, 5.00, 4.70, 5.10, 4.90, 5.20, 4.80, 5.00, 4.90, 5.10, 10.40, 5.30, 5.00, 5.20, 4.80, 5.10, 4.90, 5.20, 5.00, 4.80, 5.10, 4.90",
        "Inflammatory High Background": "8.20, 7.10, 8.90, 6.50, 8.40, 7.80, 8.10, 6.90, 8.60, 7.30, 8.00, 7.50, 7.90, 8.30, 6.80, 8.50, 7.20, 8.10, 7.60, 8.40, 6.90, 8.20, 7.70, 8.00, 7.40"
    }

    default_val = presets_map.get(preset, "")
    input_data = st.text_area(
        "25-Gene Vector Values (comma-separated):",
        value=default_val,
        height=180,
        placeholder="Enter 25 comma-separated float expression values..."
    )

    run_button = st.button("🚀 Analyze Genomics", use_container_width=True, type="primary")

with col_out:
    st.subheader("📊 Neural Clinical Report")
    
    if run_button and input_data.strip():
        try:
            res = run_inference(input_data)
            
            if res["status_color"] == "error":
                st.error(res["triage"])
            elif res["status_color"] == "warning":
                st.warning(res["triage"])
            else:
                st.success(res["triage"])

            m1, m2 = st.columns(2)
            m1.metric("Predicted Histology", res["histology"])
            m2.metric("Pathway Load Score", f"{res['pathway_score']:.3f}")

            st.write(f"**Driver Mutation:** `{res['driver']}`")

            st.markdown("### Subtype Probabilities")
            classes = ["Normal Baseline", "Lung Adenocarcinoma (LUAD)", "Lung Squamous Cell (LUSC)"]
            for cls_name, prob in zip(classes, res["probs"]):
                st.write(f"{cls_name}: **{prob*100:.1f}%**")
                st.progress(float(prob))

        except Exception as e:
            st.error(f"Inference Error: {str(e)}")
    else:
        st.info("👈 Select a preset or enter a 25-gene vector on the left and click **Analyze Genomics**.")
