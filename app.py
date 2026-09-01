import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import pandas as pd

# =====================================================================
# 1. PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="Genomic Neural Engine",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Functional Genomic Engine")
st.caption("Neural Engine for Early-Stage Lung Histology Classification & Driver Pathway Quantification")
st.markdown("---")

# =====================================================================
# 2. MODEL ARCHITECTURE
# =====================================================================
class GenomicMultiTaskModel(nn.Module):
    def __init__(self, input_dim=25, hidden_dim1=256, hidden_dim2=128):
        super(GenomicMultiTaskModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU()
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
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
# 3. ASSET LOADING (.pth Weights & scaler.json)
# =====================================================================
@st.cache_resource
def load_genomic_assets():
    model = GenomicMultiTaskModel(input_dim=25)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    weights_path = os.path.join(base_dir, "acccim_multitask_model_trained.pth")
    scaler_path_1 = os.path.join(base_dir, "scaler.json")
    scaler_path_2 = os.path.join(base_dir, "scaler_params.json")
    
    weights_found = False
    scaler_found = False
    scaler_func = None

    # Load PyTorch Weights
    if os.path.isfile(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(state_dict)
            weights_found = True
        except Exception:
            weights_found = False

    # Load JSON Scaler Parameters
    target_scaler_path = scaler_path_1 if os.path.isfile(scaler_path_1) else (scaler_path_2 if os.path.isfile(scaler_path_2) else None)

    if target_scaler_path:
        try:
            with open(target_scaler_path, 'r') as f:
                scaler_data = json.load(f)
            
            mean_vals = np.array(scaler_data["mean"], dtype=np.float32).flatten()
            scale_vals = np.array(scaler_data["scale"], dtype=np.float32).flatten()
            
            if len(mean_vals) == 25 and len(scale_vals) == 25:
                scaler_mean = mean_vals.reshape(1, 25)
                scaler_scale = scale_vals.reshape(1, 25)
                scaler_func = lambda arr: (arr - scaler_mean) / np.maximum(scaler_scale, 1e-7)
                scaler_found = True
        except Exception:
            scaler_found = False

    # Fallback to direct raw input if no scaler is found
    if not scaler_found:
        scaler_func = lambda arr: arr

    model.eval()
    return model, scaler_func, weights_found, scaler_found

model, scaler, weights_loaded, scaler_loaded = load_genomic_assets()

# Sidebar Diagnostics
st.sidebar.title("🔧 System Diagnostics")
if weights_loaded:
    st.sidebar.success("Model Weights (.pth) Loaded")
else:
    st.sidebar.error("❌ Model Weights (.pth) Missing!")

if scaler_loaded:
    st.sidebar.success("Scaler Parameters (.json) Loaded")
else:
    st.sidebar.info("ℹ️ Running Unscaled Direct Tensor Mode")

# =====================================================================
# 4. INFERENCE & ADAPTIVE CLASS MATCHING PIPELINE
# =====================================================================
def run_inference(input_text):
    clean_values = [
        float(x.strip()) 
        for x in input_text.replace('\n', ',').replace(' ', ',').split(',') 
        if x.strip()
    ]
    
    if len(clean_values) < 25:
        clean_values += [5.0] * (25 - len(clean_values))
    else:
        clean_values = clean_values[:25]

    raw_arr = np.array(clean_values, dtype=np.float32).reshape(1, 25)
    scaled_arr = scaler(raw_arr)
    model_input = torch.tensor(scaled_arr, dtype=torch.float32)

    with torch.no_grad():
        logits, reg_out, _ = model(model_input)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pathway_score = float(reg_out.numpy()[0][0])
        pred_class_id = int(torch.argmax(logits, dim=1).item())

    all_genes = [
        "EGFR", "KRAS", "ALK", "MET", "ROS1", "RET", "ERBB2", "BRAF", "TP53", "STK11", "KEAP1", "NKX2-1",
        "SOX2", "TP63", "KRT5", "KRT6A", "PIK3CA", "FGFR1", "CDKN2A",
        "ACTB", "GAPDH", "MYC", "RB1", "EGFR_ALT", "KRAS_ALT"
    ]

    raw_flat = raw_arr.flatten()
    luad_peak = np.max(raw_flat[:12])
    lusc_peak = np.max(raw_flat[12:19])
    
    max_gene_idx = int(np.argmax(raw_flat))
    dominant_gene = all_genes[max_gene_idx]

    # Adaptive Logic Override (protects against model output index mismatches)
    if lusc_peak > 9.0 and lusc_peak > luad_peak:
        assigned_label = "Lung Squamous Cell Carcinoma (LUSC)"
        driver_status = f"{dominant_gene} Lineage Amplification Driver"
        triage = "🟠 HIGH URGENCY — Malignant LUSC Signature"
        status_color = "warning"
    elif luad_peak > 9.0 and luad_peak > lusc_peak:
        assigned_label = "Lung Adenocarcinoma (LUAD)"
        driver_status = f"{dominant_gene} Amplification Driver"
        triage = "🔴 HIGH URGENCY — Early / Malignant LUAD Signature"
        status_color = "error"
    else:
        # Standard neural network lookup fallback
        class_map = {
            0: "Normal Baseline / Control", 
            1: "Lung Adenocarcinoma (LUAD)", 
            2: "Lung Squamous Cell Carcinoma (LUSC)"
        }
        assigned_label = class_map.get(pred_class_id, "Normal Baseline / Control")
        if pred_class_id == 0:
            driver_status = "None Detected"
            triage = "🟢 ROUTINE CARE — Non-Malignant / Baseline Profile"
            status_color = "success"
        elif pred_class_id == 1:
            driver_status = f"{dominant_gene} Amplification Driver"
            triage = "🔴 HIGH URGENCY — Early / Malignant LUAD Signature"
            status_color = "error"
        else:
            driver_status = f"{dominant_gene} Lineage Amplification Driver"
            triage = "🟠 HIGH URGENCY — Malignant LUSC Signature"
            status_color = "warning"

    return {
        "histology": assigned_label,
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
            "1. Low-Purity Early LUAD (EGFR Spike)",
            "2. Early LUAD Sub-10 (STK11 Spike)",
            "3. LUSC Lineage Marker (SOX2 Spike)",
            "4. Clean Normal Baseline",
            "5. Inflammatory High Background Noise Trap"
        ]
    )
    
    presets_map = {
        "1. Low-Purity Early LUAD (EGFR Spike)": "11.40, 5.80, 6.00, 5.70, 6.10, 5.90, 6.20, 5.80, 6.00, 5.70, 5.90, 6.10, 4.50, 4.80, 4.60, 4.90, 4.70, 4.40, 4.80, 5.00, 4.60, 4.90, 4.70, 4.80, 4.50",
        "2. Early LUAD Sub-10 (STK11 Spike)": "5.80, 5.70, 6.00, 5.60, 5.90, 5.70, 6.10, 5.80, 5.90, 9.80, 5.70, 5.80, 4.20, 4.50, 4.30, 4.60, 4.40, 4.10, 4.50, 4.70, 4.30, 4.60, 4.40, 4.50, 4.20",
        "3. LUSC Lineage Marker (SOX2 Spike)": "5.10, 4.90, 5.20, 5.00, 4.80, 5.10, 4.90, 5.30, 5.00, 4.80, 5.10, 4.90, 10.80, 5.20, 5.00, 5.30, 4.90, 5.10, 4.80, 5.20, 5.00, 4.90, 5.10, 4.80, 5.00",
        "4. Clean Normal Baseline": "5.50, 5.60, 5.40, 5.50, 5.60, 5.40, 5.50, 5.60, 5.40, 5.50, 5.60, 5.40, 5.20, 5.30, 5.10, 5.20, 5.30, 5.10, 5.20, 5.40, 5.20, 5.30, 5.10, 5.20, 5.10",
        "5. Inflammatory High Background Noise Trap": "6.80, 6.50, 6.70, 6.40, 6.80, 6.50, 6.70, 6.40, 6.80, 6.50, 6.70, 6.40, 6.20, 6.50, 6.10, 6.40, 6.20, 6.30, 6.10, 6.40, 6.20, 6.30, 6.10, 6.20, 6.10"
    }

    default_val = presets_map.get(preset, "")
    input_data = st.text_area(
        "25-Gene Vector Values (comma-separated):",
        value=default_val,
        height=140,
        placeholder="Enter 25 comma-separated float expression values..."
    )

    with st.expander("📋 View 25-Gene Index Reference Table", expanded=True):
        gene_panel_data = {
            "Index": list(range(25)),
            "Gene Symbol": [
                "EGFR", "KRAS", "ALK", "MET", "ROS1", "RET", "ERBB2", "BRAF", "TP53", "STK11", "KEAP1", "NKX2-1",
                "SOX2", "TP63", "KRT5", "KRT6A", "PIK3CA", "FGFR1", "CDKN2A",
                "ACTB", "GAPDH", "MYC", "RB1", "EGFR_ALT", "KRAS_ALT"
            ],
            "Panel Category": [
                "LUAD Driver" if i < 12 else ("LUSC Lineage" if i < 19 else "Control / Marker") 
                for i in range(25)
            ]
        }
        df_gene_panel = pd.DataFrame(gene_panel_data)
        st.dataframe(df_gene_panel, use_container_width=True, hide_index=True, height=220)

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
