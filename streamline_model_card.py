# streamline_model_card.py
"""
Streamlit app: Streamline Model Card Generator

Features:
- Connect to MLflow (optional, best-effort) and pick latest run
- Upload metadata JSON (artifact metadata) or use simulated metadata
- Use Model Card Toolkit if installed to produce structured card (best-effort)
- Generate Markdown and HTML model cards, preview them and download
- Simple fairness / metric display and warnings about sensitive features

Run:
    pip install streamlit markdown mlflow model-card-toolkit joblib
    streamlit run streamline_model_card.py

Notes:
- MLflow connection is "best-effort": set MLFLOW_TRACKING_URI env var if you want to connect to a remote server.
- Model Card Toolkit usage is optional and performed if available.
"""
import os
import json
from datetime import datetime
from io import BytesIO
import streamlit as st

# optional libs
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

try:
    from model_card_toolkit import ModelCardToolkit
    MCT_AVAILABLE = True
except Exception:
    MCT_AVAILABLE = False

# helper: default simulated metadata (same structure used earlier)
def default_metadata():
    return {
        "model_name": "income-prediction-logreg",
        "model_version": "v1.2.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "training_data": {
            "name": "UCI Adult (preprocessed) / Synthetic fallback",
            "reference_csv": "data/reference.csv",
            "current_csv": "data/current.csv",
            "num_examples": 5000,
            "features": ["age", "education_num", "hours_per_week", "race_A", "race_B", "race_C", "race_D"]
        },
        "training_procedure": {
            "framework": "scikit-learn LogisticRegression (fallback)",
            "hyperparameters": {"C": 1.0, "max_iter": 500},
            "dp": {"used": False},
            "federated_simulation": {"clients": 3, "rounds": 2}
        },
        "evaluation": {
            "dataset": "current.csv (holdout)",
            "metrics": {"accuracy": 0.85, "precision": 0.78, "recall": 0.62},
            "fairness_checks": {
                "race_parity_gap": {"A_vs_D_accuracy_diff": 0.07},
                "notes": "Basic group checks only; no mitigation applied."
            }
        },
        "artifacts": ["models/sklearn_model.joblib", "models/ct_transformer.joblib"],
        "provenance": {
            "git_commit": "abc123def (simulated)",
            "run_command": "python privacy_pipeline_improved.py --train",
            "owner": "ml-team@example.com"
        },
        "intended_use": "Predict whether income >50K given demographic and employment features. For research and internal analytics only.",
        "limitations": [
            "Model trained on preprocessed features; upstream preprocessing must be identical at inference.",
            "Sensitive correlations exist (race proxies) — privacy & fairness review recommended."
        ],
        "license": "Apache-2.0",
        "security": "Model does not include credentials; artifacts stored on internal storage."
    }

# small utility: produce markdown model card from metadata (best-effort)
def generate_markdown(metadata: dict) -> str:
    def md_table(d):
        if not d:
            return ""
        lines = ["| Key | Value |", "|---|---|"]
        for k, v in d.items():
            val = json.dumps(v) if not isinstance(v, str) else v
            lines.append(f"| {k} | `{val}` |")
        return "\n".join(lines) + "\n\n"

    lines = []
    lines.append(f"# Model Card — {metadata.get('model_name','Unnamed Model')}")
    lines.append("")
    lines.append(f"**Version:** {metadata.get('model_version','v0')}  ")
    lines.append(f"**Created:** {metadata.get('created_at')}  ")
    lines.append("")
    lines.append("## Summary")
    lines.append(metadata.get("intended_use", "No summary provided."))
    lines.append("")
    lines.append("## Model Details")
    lines.append(f"- **Framework:** {metadata.get('training_procedure',{}).get('framework')}")
    dp_used = metadata.get('training_procedure',{}).get('dp',{}).get('used', False)
    lines.append(f"- **DP used:** {dp_used}")
    lines.append(f"- **Artifacts:** {', '.join(metadata.get('artifacts',[]))}")
    lines.append("")
    lines.append("## Training Data")
    td = metadata.get("training_data", {})
    lines.append(f"- **Dataset name:** {td.get('name')}  ")
    lines.append(f"- **Examples:** {td.get('num_examples')}  ")
    lines.append(f"- **Features (sample):** {', '.join(td.get('features',[])[:10])}  ")
    lines.append("")
    lines.append("## Training Procedure & Hyperparameters")
    hp = metadata.get("training_procedure",{}).get("hyperparameters",{})
    if hp:
        lines.append("| Hyperparameter | Value |")
        lines.append("|---|---|")
        for k,v in hp.items():
            lines.append(f"| {k} | `{v}` |")
        lines.append("")
    lines.append("## Evaluation")
    evalm = metadata.get("evaluation",{})
    if evalm.get("metrics"):
        lines.append("### Metrics")
        for k,v in evalm.get("metrics",{}).items():
            lines.append(f"- **{k}**: `{v}`  ")
    if evalm.get("fairness_checks"):
        lines.append("")
        lines.append("### Fairness / Checks")
        for k,v in evalm.get("fairness_checks",{}).items():
            lines.append(f"- **{k}**: `{v}`  ")
    lines.append("")
    lines.append("## Intended Use")
    lines.append(metadata.get("intended_use",""))
    lines.append("")
    lines.append("## Limitations & Risks")
    for item in metadata.get("limitations",[]):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Reproducibility & Provenance")
    prov = metadata.get("provenance",{})
    if prov:
        lines.append(md_table(prov))
    lines.append("## Security & Privacy")
    lines.append(metadata.get("security",""))
    lines.append("")
    lines.append("## License")
    lines.append(metadata.get("license",""))
    lines.append("")
    lines.append("## How to cite")
    lines.append(f"- Model: `{metadata.get('model_name')}` version `{metadata.get('model_version')}`")
    return "\n".join(lines)

# Streamlit UI
st.set_page_config(page_title="Streamline — Model Card Generator", layout="wide")
st.title("Streamline — Model Card Generator")

with st.sidebar:
    st.header("Source")
    source = st.radio("Select metadata source", options=["Simulated", "Upload JSON", "MLflow (best-effort)"])
    if source == "Upload JSON":
        uploaded = st.file_uploader("Upload metadata JSON", type=["json"], accept_multiple_files=False)
    elif source == "MLflow (best-effort)":
        st.write("MLflow available:" , "✅" if MLFLOW_AVAILABLE else "❌")
        mlflow_experiment = st.text_input("Experiment name or ID (leave blank for Default)", value="")
        mlflow_fetch = st.button("Fetch latest run from MLflow")
    st.markdown("---")
    st.header("Options")
    use_mct = st.checkbox("Use Model Card Toolkit (if available)", value=False)
    st.markdown("Downloads:")
    st.write("Markdown and HTML model cards will be available after generation.")

# Fetch or load metadata
metadata = None
if source == "Simulated":
    metadata = default_metadata()
    st.sidebar.success("Using simulated metadata")
elif source == "Upload JSON":
    if uploaded:
        try:
            metadata = json.load(uploaded)
            st.sidebar.success("Loaded uploaded metadata")
        except Exception as e:
            st.sidebar.error(f"Failed to parse JSON: {e}")
            metadata = default_metadata()
    else:
        st.sidebar.info("Upload a JSON file or switch to Simulated")
        metadata = default_metadata()
elif source == "MLflow (best-effort)":
    if not MLFLOW_AVAILABLE:
        st.sidebar.error("MLflow client not installed or not available. Install `mlflow` or pick another source.")
        metadata = default_metadata()
    else:
        # attempt to fetch latest run
        try:
            client = mlflow.tracking.MlflowClient()
            # choose experiment
            if mlflow_experiment:
                try:
                    exp = client.get_experiment_by_name(mlflow_experiment)
                    if exp is None:
                        exp = client.get_experiment(mlflow_experiment)
                except Exception:
                    exp = client.get_experiment_by_name(mlflow_experiment)
            else:
                exp = client.get_experiment_by_name("Default") or client.get_experiment(0)
            if exp is None:
                st.sidebar.error("Experiment not found; using simulated metadata.")
                metadata = default_metadata()
            else:
                runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
                if runs:
                    run = runs[0]
                    metadata = {
                        "model_name": run.data.tags.get("mlflow.runName", "mlflow-run"),
                        "model_version": run.info.run_id,
                        "created_at": datetime.utcfromtimestamp(int(run.info.start_time)/1000).isoformat()+"Z",
                        "training_data": {"name": run.data.tags.get("training_dataset", "unknown")},
                        "training_procedure": {"framework": run.data.tags.get("framework","unknown"), "hyperparameters": dict(run.data.params)},
                        "evaluation": {"metrics": dict(run.data.metrics)},
                        "artifacts": [a.path for a in client.list_artifacts(run.info.run_id)],
                        "provenance": {"run_id": run.info.run_id, "owner": run.data.tags.get("user","unknown")},
                        "intended_use": run.data.tags.get("intended_use", ""),
                        "limitations": run.data.tags.get("limitations","").split(";") if run.data.tags.get("limitations") else []
                    }
                    st.sidebar.success("Fetched run from MLflow")
                else:
                    st.sidebar.warning("No runs found in experiment; using simulated metadata")
                    metadata = default_metadata()
        except Exception as e:
            st.sidebar.error(f"MLflow fetch failed: {e}")
            metadata = default_metadata()

# Show quick metadata summary
st.subheader("Metadata preview")
col1, col2 = st.columns([2,1])
with col1:
    st.json(metadata)
with col2:
    st.metric("Model", metadata.get("model_name","-"))
    st.metric("Version", metadata.get("model_version","-"))
    if metadata.get("evaluation",{}).get("metrics"):
        m = metadata["evaluation"]["metrics"]
        st.metric("Accuracy", m.get("accuracy","-"))
        st.metric("Precision", m.get("precision","-"))

st.markdown("---")
st.subheader("Generate model card")

if st.button("Generate Model Card"):
    # Use MCT if requested & available
    md = None
    html = None
    try:
        if use_mct and MCT_AVAILABLE:
            st.info("Using Model Card Toolkit to build structured model card (best-effort)...")
            outdir = os.path.join(os.getcwd(), "mct_output")
            os.makedirs(outdir, exist_ok=True)
            toolkit = ModelCardToolkit(outdir)
            try:
                _ = toolkit.scaffold_assets(model_name=metadata.get("model_name","unnamed"))
                card = toolkit.model_card
                card.model_details.name = metadata.get("model_name","unnamed")
                card.model_details.version = metadata.get("model_version","v0")
                card.model_details.overview = metadata.get("intended_use","")
                # fill some fields
                card.model_parameters.training_data = metadata.get("training_data",{})
                card.model_parameters.quantitative_analysis = {"evaluation_metrics": metadata.get("evaluation",{}).get("metrics",{})}
                # export markdown + html
                toolkit.export_format(output_fname="model_card_mct", output_path=outdir, output_formats=["markdown","html"])
                md_path = os.path.join(outdir, "model_card_mct.md")
                html_path = os.path.join(outdir, "model_card_mct.html")
                if os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        md = f.read()
                if os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        html = f.read()
            except Exception as e:
                st.warning(f"Model Card Toolkit integration failed: {e}. Falling back to manual generator.")
                md = generate_markdown(metadata)
        else:
            md = generate_markdown(metadata)
    except Exception as e:
        st.error(f"Generation failed: {e}")
        md = generate_markdown(metadata)

    # ensure md content
    if md is None:
        md = generate_markdown(metadata)

    # Build HTML from markdown
    try:
        import markdown as mdlib
        html_body = mdlib.markdown(md, extensions=["tables", "fenced_code"])
        html = f"<html><head><meta charset='utf-8'><title>Model Card</title></head><body>{html_body}</body></html>"
    except Exception:
        html = "<html><body><pre>" + md.replace("<","&lt;") + "</pre></body></html>"

    # show previews and download buttons
    st.success("Model card generated.")
    st.subheader("Markdown Preview")
    st.markdown(md, unsafe_allow_html=True)

    st.subheader("HTML Preview")
    st.components.v1.html(html, height=600, scrolling=True)

    # prepare downloads
    md_bytes = md.encode("utf-8")
    html_bytes = html.encode("utf-8")
    st.download_button("Download Markdown (.md)", data=md_bytes, file_name=f"{metadata.get('model_name','model_card')}.md", mime="text/markdown")
    st.download_button("Download HTML (.html)", data=html_bytes, file_name=f"{metadata.get('model_name','model_card')}.html", mime="text/html")

    # Extra: quick checks (sensitive features / DP flag)
    st.markdown("---")
    st.subheader("Quick checks & guidance")
    features = metadata.get("training_data",{}).get("features",[])
    sens_keys = [f for f in features if any(tok in f.lower() for tok in ["race","sex","gender","religion","ethnicity"])]
    if sens_keys:
        st.warning(f"Detected potentially sensitive features: {', '.join(sens_keys)}. Consider fairness evaluation and privacy controls.")
    if metadata.get("training_procedure",{}).get("dp",{}).get("used"):
        st.info("Model was trained with DP enabled (per metadata). Confirm epsilon and auditing info in model card.")
    else:
        st.info("DP not used (per metadata). If privacy is a concern, consider retraining with DP or applying differential privacy pipeline.")

st.markdown("---")
st.caption("Streamline — Model Card Generator. Designed to quickly turn artifact metadata into downloadable model cards. Integrates MLflow & Model Card Toolkit when available.")

