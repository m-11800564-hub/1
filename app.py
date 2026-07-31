import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="ACCCIM Neural Engine", page_icon="🧬")

# 1. Model Architecture matching your trained weights
class ACCCIMMultiTaskModel(nn.Module):
    def __init__(self, input_dim=25, hidden_dim1=256, hidden_dim2=128):
        super(ACCCIMMultiTaskModel, self).__init__()
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

# 2. Load Model
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
    st.error(f"Error loading model file: {e}")

# 3. User Interface
st.title("🧬 ACCCIM Neural Engine")
st.subheader("Diagnostic Report & Clinical Triage Dashboard")

input_text = st.text_area(
    "Paste 25 Raw Gene Expression Log-Counts:",
    value="2.1, 1.5, 0.4, 0.8, 1.1, 19.8, 0.5, 0.3, 1.2, 0.4, 0.2, 0.9, 0.1, 0.4, 0.7, 0.3, 0.5, 0.2, 0.1, 0.8, 0.4, 0.3, 0.6, 0.2, 0.1",
    height=120
)

if st.button("Run Model Inference", type="primary"):
    if not model_ready:
        st.error("Model weights are missing or invalid!")
        st.stop()
        
    try:
        clean_values = [float(x.strip()) for x in input_text.replace('\n', ',').replace(' ', ',').split(',') if x.strip()]
        if len(clean_values) < 25:
            clean_values += [0.0] * (25 - len(clean_values))
        else:
            clean_values = clean_values[:25]

        raw_tensor = torch.tensor(clean_values, dtype=torch.float32)
        log_tensor = torch.log2(raw_tensor + 1.0)
        mean, std = log_tensor.mean(), log_tensor.std() + 1e-6
        z_score_tensor = (log_tensor - mean) / std
        model_input = z_score_tensor.unsqueeze(0)

        with torch.no_grad():
            logits, reg_out, _ = model(model_input)
            probs = F.softmax(logits, dim=1).numpy()[0]
            pathway_score = float(reg_out.numpy()[0][0])
            pred_class_id = int(torch.argmax(logits, dim=1).item())

        class_map = {0: "Normal / Benign Baseline", 1: "Lung Adenocarcinoma (LUAD)", 2: "Lung Squamous Cell (LUSC)"}

        st.success("PyTorch Forward Pass Completed Successfully!")
        st.write(f"**Predicted Histology:** {class_map[pred_class_id]}")
        st.write(f"**Confidence:** {probs[pred_class_id]*100:.2f}%")
        st.write(f"**Driver Pathway Load Score:** {pathway_score:.3f}")

    except Exception as e:
        st.error(f"Inference Error: {e}")
