import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="ACCCIM Neural Engine", page_icon="🧬")

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
# 3. STREAMLIT UI & INTERFACE
# =====================================================================
st.title("🧬 ACCCIM Neural Engine")
st.subheader("Diagnostic Report & Clinical Triage Dashboard")

# Interactive Preset Buttons
st.markdown("### 🧪 Quick Preset Validation Samples")
col1, col2, col3 = st.columns(3)

default_text = "2.1, 1.5, 0.4, 0.8, 1.1, 19.8, 0.5, 0.3, 1.2, 0.4, 0.2, 0.9, 0.1, 0.4, 0.7, 0.3, 0.5, 0.2, 0.1, 0.8, 0.4, 0.3, 0.6, 0.2, 0.1"

if col1.button("Load Normal Sample"):
    default_text = "1.1, 0.9, 0.4, 0.8, 1.0, 0.5, 0.3, 0.6, 1.1, 0.4, 0.2, 0.7, 0.1, 0.4, 0.5, 0.3, 0.2, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1"

if col2.button("Load LUAD Sample"):
    default_text = "2.1, 1.5, 0.4, 0.8, 1.1, 19.8, 0.5, 0.3, 1.2, 0.4, 0.2, 0.9, 0.1, 0.4, 0.7, 0.3, 0.5, 0.2, 0.1, 0.8, 0.4, 0.3, 0.6, 0.2, 0.1"

if col3.button("Load LUSC Sample"):
    default_text = "0.2, 0.4, 18.5, 17.2, 0.3, 0.5, 0.1, 0.4, 0.2, 0.6, 0.3, 0.1, 0.5, 0.2, 0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2"

input_text = st.text_area(
    "Paste 25 Raw Gene Expression Log-Counts (Comma-Separated):",
    value=default_text,
    height=100
)

# =====================================================================
# 4. INFERENCE PIPELINE
# =====================================================================
if st.button("Run Model Inference", type="primary"):
    if not model_ready:
        st.error("Model weights are missing or invalid!")
        st.stop()
        
    try:
        # Parse inputs
        clean_values = [float(x.strip()) for x in input_text.replace('\n', ',').replace(' ', ',').split(',') if x.strip()]
        if len(clean_values) < 25:
            clean_values += [0.0] * (25 - len(clean_values))
        else:
            clean_values = clean_values[:25]

        # Convert to Tensor (Shape: [1, 25])
        raw_tensor = torch.tensor(clean_values, dtype=torch.float32).unsqueeze(0)

        # Forward Pass
        with torch.no_grad():
            logits, reg_out, _ = model(raw_tensor)
            probs = F.softmax(logits, dim=1).numpy()[0]
            pathway_score = float(reg_out.numpy()[0][0])
            pred_class_id = int(torch.argmax(logits, dim=1).item())

        class_map = {
            0: "Normal / Benign Baseline", 
            1: "Lung Adenocarcinoma (LUAD)", 
            2: "Lung Squamous Cell (LUSC)"
        }

        # Display Results
        st.success("PyTorch Forward Pass Completed Successfully!")
        st.write(f"**Predicted Histology:** {class_map[pred_class_id]}")
        st.write(f"**Confidence:** {probs[pred_class_id]*100:.2f}%")
        st.write(f"**Driver Pathway Load Score:** {pathway_score:.3f}")

        # Triage Badge
        if pred_class_id == 0:
            st.info("🟢 ROUTINE CARE — Non-Malignant / Baseline Genomic Profile")
        elif pred_class_id == 1:
            st.error("🔴 CRITICAL URGENCY — Malignant LUAD Signature (Route to Thoracic Oncology)")
        else:
            st.warning("🟠 HIGH URGENCY — Malignant LUSC Signature (Route for Immunotherapy Evaluation)")

    except Exception as e:
        st.error(f"Inference Error: {e}")
