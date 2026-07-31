import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(page_title="ACCCIM Neural Engine", page_icon="🧬")

# 1. DEFINE MODEL ARCHITECTURE
class GenomicClassifier(nn.Module):
    def __init__(self):
        super(GenomicClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.encoder(x)

# 2. LOAD MODEL WEIGHTS
@st.cache_resource
def load_model():
    model = GenomicClassifier()
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

# 3. UI DASHBOARD
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
            logits = model(model_input)
            prediction_idx = torch.argmax(logits, dim=1).item()

        st.success("PyTorch Forward Pass Completed Successfully!")
        st.write(f"**Predicted Class Index:** {prediction_idx}")
        st.write(f"**Output Logits:** {logits.numpy().tolist()}")

    except Exception as e:
        st.error(f"Inference Error: {e}")
