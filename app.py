# =====================================================================
# 4. INFERENCE & DUAL-HEAD DISPLAY PIPELINE
# =====================================================================
if st.button("Run Model Inference", type="primary"):
    if not model_ready:
        st.error("Model weights are missing or invalid!")
        st.stop()
        
    try:
        # 1. Parse raw gene log-counts from user input
        clean_values = [float(x.strip()) for x in input_text.replace('\n', ',').replace(' ', ',').split(',') if x.strip()]
        if len(clean_values) < 25:
            clean_values += [1.0] * (25 - len(clean_values))  # Neutral baseline padding
        else:
            clean_values = clean_values[:25]

        # 2. EXACT COLAB NORMALIZATION PIPELINE
        raw_arr = np.array(clean_values, dtype=np.float32)
        raw_tensor = torch.tensor(raw_arr, dtype=torch.float32)
        
        # Log2 Transform + Sample-level Z-Score (Matches Colab)
        log_tensor = torch.log2(raw_tensor + 1.0)
        std_val = log_tensor.std()
        # Handle zero standard deviation (e.g., if all input values are identical)
        std_safe = std_val + 1e-6 if std_val != 0 else 1.0
        z_score_tensor = (log_tensor - log_tensor.mean()) / std_safe
        
        # Add batch dimension: Shape -> (1, 25)
        model_input = z_score_tensor.unsqueeze(0)

        # 3. Model Forward Pass
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
        
        # Explicit Head Outputs
        st.code(
            f"• Histology Head       : {class_map[pred_class_id]}\n"
            f"• Driver Mutation Head : {driver_head_status}",
            language="text"
        )

        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.write(f"**Top Class Confidence:** {probs[pred_class_id]*100:.2f}%")
            st.write(f"**Driver Pathway Load Score:** {pathway_score:.3f}")
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
