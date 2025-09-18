#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import json, math, io, random, zipfile
import statistics as stats
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="R&D MD and Analysis",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    /* Subheader styling */
    .custom-subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2a5298;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2a5298;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Section dividers */
    .section-divider {
        margin: 2rem 0;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a5298;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Generic utilities (unchanged)
# =========================

def parse_values(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            try:
                out.append(float(item))
            except Exception:
                if isinstance(item, dict):
                    for k in ("value","intensity","y"):
                        if k in item:
                            try:
                                out.append(float(item[k]))
                            except Exception:
                                pass
        return out
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                return [float(x) for x in arr]
            except Exception:
                pass
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except Exception:
            return []
    return []

def robust_mean(values):
    vals = [float(v) for v in values if v is not None]
    if len(vals) == 0:
        return float("nan"), []
    med = stats.median(vals)
    mad = stats.median([abs(x - med) for x in vals])
    if mad == 0:
        return sum(vals)/len(vals), vals
    thr = 3.0 * 1.4826 * mad
    kept = [x for x in vals if abs(x - med) <= thr]
    if not kept:
        kept = vals
    return sum(kept)/len(kept), kept

def _find_scans(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "scans" and isinstance(v, list):
                return v
            found = _find_scans(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_scans(item)
            if found is not None:
                return found
    return None

def _get_channel_array(node, key):
    if key in node:
        return parse_values(node.get(key))
    for k in list(node.keys()):
        if isinstance(k, str) and k.lower() == key.lower():
            return parse_values(node.get(k))
    for alt in ["channels","Channels","data","Data"]:
        if isinstance(node.get(alt), dict):
            m = node[alt]
            if key in m:
                return parse_values(m[key])
            for mk in m.keys():
                if isinstance(mk,str) and mk.lower() == key.lower():
                    return parse_values(m[mk])
    return []

def extract_bg_sample(obj, chan_key="SC_Green"):
    scans = _find_scans(obj) or []
    bg, sm = [], []
    for node in scans:
        if not isinstance(node, dict):
            continue
        params = node.get("parameters") or {}
        stype = params.get("scanType") or node.get("scanType") or ""
        vals = _get_channel_array(node, chan_key)
        if not vals:
            continue
        s = str(stype).lower()
        if s.startswith("back"):
            bg = vals
        elif s.startswith("sam"):
            sm = vals
    return bg, sm

def extract_sample_node(obj):
    scans = _find_scans(obj) or []
    for node in scans:
        if not isinstance(node, dict):
            continue
        params = node.get("parameters") or {}
        stype = params.get("scanType") or node.get("scanType") or ""
        if str(stype).lower().startswith("sam"):
            return node
    return None

def get_loc_doses_from_sample(obj):
    node = extract_sample_node(obj)
    doses = {}
    if node is None:
        return doses
    for src in [node, node.get("parameters", {})]:
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(k, str) and k.upper().startswith("LOC"):
                    try:
                        val = float(v)
                        if abs(val) > 0:
                            doses[k] = val
                    except Exception:
                        pass
    return doses

def compute_absorbance(file_bytes, chan_key="SC_Green", formula="log10_bg_over_sample"):
    try:
        obj = json.loads(file_bytes.decode("utf-8"))
    except Exception:
        obj = json.loads(file_bytes)
    bg, sm = extract_bg_sample(obj, chan_key=chan_key)
    if not bg or not sm:
        raise ValueError("Missing Background or Sample intensities for the selected channel.")
    bg_mean, _ = robust_mean(bg)
    sm_mean, _ = robust_mean(sm)
    if bg_mean <= 0 or sm_mean <= 0:
        raise ValueError("Non-positive intensities.")
    if formula == "log10_bg_over_sample":
        A = math.log10(bg_mean / sm_mean)
    elif formula == "absorbance_single":
        A = -math.log10(sm_mean / bg_mean)
    else:
        A = math.log10(bg_mean / sm_mean)
    diag = {"bg_avg": bg_mean, "sm_avg": sm_mean, "bg_n": len(bg), "sm_n": len(sm)}
    return A, diag, obj

def compute_spike_conc_mgL(stock_mgL, spike_uL, base_sample_mL=40.0, extra_mL=0.0):
    V_spike_mL = spike_uL / 1000.0
    V_total_mL = base_sample_mL + extra_mL
    if V_total_mL <= 0:
        return float("nan")
    return stock_mgL * (V_spike_mL / V_total_mL)

def fit_linear(xs, ys, weights=None):
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    w = np.ones_like(xs) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != xs.shape: w = np.ones_like(xs)
    W = np.sum(w); xw = np.sum(w*xs)/W; yw = np.sum(w*ys)/W
    num = np.sum(w*(xs-xw)*(ys-yw)); den = np.sum(w*(xs-xw)**2)
    if den == 0: raise ValueError("Zero variance in x.")
    m = num/den; b = yw - m*xw
    ss_tot = np.sum(w*(ys-yw)**2); ss_res = np.sum(w*(ys-(m*xs+b))**2)
    R2 = 1.0 - (ss_res/ss_tot if ss_tot>0 else 0.0)
    return float(m), float(b), float(R2)

def fit_quadratic(xs, ys, weights=None):
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    X = np.vstack([xs**2, xs, np.ones_like(xs)]).T
    if weights is None:
        W = np.eye(len(xs))
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != xs.shape: w = np.ones_like(xs)
        W = np.diag(w)
    try:
        beta = np.linalg.inv(X.T@W@X)@(X.T@W@ys)
    except np.linalg.LinAlgError:
        raise ValueError("Quadratic fit failed (singular).")
    yhat = X@beta
    ss_tot = np.sum((ys - np.mean(ys))**2)
    ss_res = np.sum((ys - yhat)**2)
    R2 = 1 - (ss_res/ss_tot if ss_tot>0 else 0.0)
    return float(beta[0]), float(beta[1]), float(beta[2]), float(R2)

def lod_loq(blank_As, slope):
    if slope == 0 or not blank_As: return float("nan"), float("nan"), 0.0
    sd = float(np.std(blank_As, ddof=0))
    return 3.3*sd/abs(slope), 10.0*sd/abs(slope), sd

def predict_conc_linear(A, m, b): return (A - b)/m

def plot_calibration(xs, ys, m=None, b=None, quad=None, title="Calibration", xlabel="Conc (mg/L)", ylabel="Absorbance (A)"):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(xs, ys, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    if len(xs)>=2:
        x_min, x_max = min(xs), max(xs); span = x_max - x_min
        grid_x = np.linspace(x_min-0.05*span, x_max+0.05*span if span>0 else x_max+1, 200)
        if quad is not None:
            a,b2,c = quad
            ax.plot(grid_x, a*grid_x**2 + b2*grid_x + c, 'r-', linewidth=2, label='Quadratic fit')
        elif m is not None and b is not None:
            ax.plot(grid_x, m*grid_x + b, 'b-', linewidth=2, label='Linear fit')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    if m is not None or quad is not None:
        ax.legend()
    fig.tight_layout()
    png = io.BytesIO(); fig.savefig(png, format="png", dpi=300, bbox_inches="tight"); png.seek(0)
    pdf = io.BytesIO(); fig.savefig(pdf, format="pdf", bbox_inches="tight"); pdf.seek(0)
    plt.close(fig); return png.getvalue(), pdf.getvalue()

# Residuals, LOOCV, and session I/O
def calc_residuals(xs, ys, model):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if model.get("model") == "linear":
        m = model["m"]; b = model["b"]
        yhat = m*xs + b
    else:
        a,b2,c = model["a"], model["b"], model["c"]
        yhat = a*xs**2 + b2*xs + c
    resid = ys - yhat
    return yhat, resid

def loocv_metrics(xs, ys, model_type="linear"):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    n = len(xs)
    if n < 3:
        return {"n": n, "PRESS": float("nan"), "RMSE_LOO": float("nan"), "R2_LOO": float("nan")}
    yhat_loo = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        xtr, ytr = xs[mask], ys[mask]
        try:
            if model_type == "linear":
                m,b,_ = fit_linear(xtr, ytr)
                yhat_loo[i] = m*xs[i] + b
            else:
                a,b2,c,_ = fit_quadratic(xtr, ytr)
                yhat_loo[i] = a*xs[i]**2 + b2*xs[i] + c
        except Exception:
            yhat_loo[i] = np.nan
    resid = ys - yhat_loo
    press = float(np.nansum(resid**2))
    rmse = float(np.sqrt(press / np.sum(~np.isnan(resid))))
    ss_tot = float(np.nansum((ys - np.nanmean(ys))**2))
    r2_loo = float(1 - (press/ss_tot if ss_tot>0 else np.nan))
    return {"n": int(n), "PRESS": press, "RMSE_LOO": rmse, "R2_LOO": r2_loo}

def plot_residuals(xs, ys, model, title="Residuals"):
    yhat, resid = calc_residuals(xs, ys, model)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(yhat, resid, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax.axhline(0, linestyle="--", color='red', alpha=0.7)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Fitted values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = io.BytesIO(); fig.savefig(png, format="png", dpi=300, bbox_inches="tight"); png.seek(0)
    pdf = io.BytesIO(); fig.savefig(pdf, format="pdf", bbox_inches="tight"); pdf.seek(0)
    plt.close(fig); return png.getvalue(), pdf.getvalue()

def export_session_zip(analyte, profile, cal_table=None, model=None, plots=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("profile.json", json.dumps(profile, indent=2))
        manifest = {"analyte": analyte, "has_cal_table": cal_table is not None, "has_model": model is not None, "plots": list(plots.keys()) if plots else []}
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        if cal_table is not None:
            z.writestr("calibration.csv", cal_table.to_csv(index=False))
        if model is not None:
            z.writestr("model.json", json.dumps(model, indent=2))
        if plots:
            for name, bytes_data in plots.items():
                z.writestr(f"plots/{name}", bytes_data)
    buf.seek(0)
    return buf.getvalue()

def import_session_zip(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        manifest = json.loads(z.read("manifest.json").decode("utf-8")) if "manifest.json" in z.namelist() else {}
        profile = json.loads(z.read("profile.json").decode("utf-8")) if "profile.json" in z.namelist() else None
        cal_table = None
        if "calibration.csv" in z.namelist():
            from io import StringIO
            cal_table = pd.read_csv(StringIO(z.read("calibration.csv").decode("utf-8")))
        model = json.loads(z.read("model.json").decode("utf-8")) if "model.json" in z.namelist() else None
        plots = {}
        for name in z.namelist():
            if name.startswith("plots/") and name.lower().endswith(".png"):
                plots[name.split("/",1)[1]] = z.read(name)
    return {"manifest": manifest, "profile": profile, "cal_table": cal_table, "model": model, "plots": plots}

# Fitting helpers
def residuals_linear(xs, ys, m, b):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    yhat = m*xs + b
    return (ys - yhat).tolist(), yhat.tolist()

def loocv_linear(xs, ys, weights=None):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    n = len(xs)
    errs = []
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        try:
            m_i, b_i, _ = fit_linear(xs[mask], ys[mask], None if weights is None else np.asarray(weights)[mask])
            y_pred = m_i*xs[i] + b_i
            errs.append(float(ys[i] - y_pred))
        except Exception:
            errs.append(float("nan"))
    fe = np.array([e for e in errs if np.isfinite(e)])
    rmse = float(np.sqrt(np.mean(fe**2))) if fe.size else float("nan")
    mae = float(np.mean(np.abs(fe))) if fe.size else float("nan")
    return {"errors": errs, "RMSE": rmse, "MAE": mae}

def make_residual_plot(xs, res, title="Residuals vs Conc", xlabel="Conc (mg/L)"):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(xs, res, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax.axhline(0, linestyle="--", color='red', alpha=0.7)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Residual (A)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight"); buf.seek(0); plt.close(fig)
    return buf.getvalue()

def export_project_zip(analyte, cal_df, fit_json, cal_png=None, cal_pdf=None, residual_png=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if cal_df is not None:
            z.writestr(f"{analyte}_calibration_table.csv", cal_df.to_csv(index=False))
        if fit_json is not None:
            z.writestr(f"{analyte}_model.json", json.dumps(fit_json, indent=2))
        if cal_png is not None:
            z.writestr(f"{analyte}_calibration.png", cal_png)
        if cal_pdf is not None:
            z.writestr(f"{analyte}_calibration.pdf", cal_pdf)
        if residual_png is not None:
            z.writestr(f"{analyte}_residuals.png", residual_png)
    buf.seek(0)
    return buf.getvalue()

# Profiles (per-analyte)
DEFAULT_PROFILE = {
    "analyte": "Iron",
    "channel": "SC_Green",
    "absorbance_formula": "log10_bg_over_sample",
    "defaults": {"base_sample_mL": 40.0, "extra_constant_mL": 0.0, "include_all_loc_volumes": True},
    "locs": {}
}

def get_profile_store():
    if "profiles" not in st.session_state:
        st.session_state["profiles"] = {}
    return st.session_state["profiles"]

def get_profile(analyte_name):
    store = get_profile_store()
    if analyte_name not in store:
        prof = json.loads(json.dumps(DEFAULT_PROFILE))
        prof["analyte"] = analyte_name
        store[analyte_name] = prof
    return store[analyte_name]

def set_profile(analyte_name, prof):
    store = get_profile_store(); store[analyte_name] = prof

def profile_pick_standard(prof, detected_locs: dict):
    loc_map = prof.get("locs", {})
    candidates = [k for k in detected_locs.keys() if loc_map.get(k, {}).get("role") == "standard"]
    if not candidates:
        return None, None
    candidates.sort(key=lambda k: float(detected_locs.get(k, 0.0)), reverse=True)
    chosen = candidates[0]
    stock = float(loc_map.get(chosen, {}).get("stock_mgL", 0.0) or 0.0)
    if stock <= 0: return chosen, None
    return chosen, stock

# DOE Templates
DOE_TEMPLATES = {
    "Iron": {
        "screening": "pH: 3, 6\nBuffer_mM: 10, 50\nPhenanthroline_mM: 0.5, 2.0\nReactionTime_min: 1, 10\nTemperature_C: 20, 35",
        "ccd": "pH: 3, 6\nBuffer_mM: 10, 50\nPhenanthroline_mM: 0.5, 2.0\nReactionTime_min: 1, 10"
    },
    "Phosphate": {
        "screening": "pH: 1, 2\nMolybdate_mM: 5, 20\nAscorbicAcid_mM: 10, 50\nReactionTime_min: 5, 15\nTemperature_C: 20, 35",
        "ccd": "pH: 1, 2\nMolybdate_mM: 5, 20\nAscorbicAcid_mM: 10, 50\nReactionTime_min: 5, 15"
    },
    "Ammonia": {
        "screening": "pH: 10, 11.5\nNessler_mM: 1, 5\nReactionTime_min: 5, 20\nTemperature_C: 20, 35",
        "ccd": "pH: 10, 11.5\nNessler_mM: 1, 5\nReactionTime_min: 5, 20"
    }
}

# DOE Functions
def full_factorial(levels_dict):
    keys = list(levels_dict.keys())
    grids = [[]]
    for k in keys:
        new = []
        for row in grids:
            for v in levels_dict[k]:
                new.append(row + [v])
        grids = new
    rows = []
    for i, comb in enumerate(grids, start=1):
        d = {"run": i}
        for k, v in zip(keys, comb): d[k] = v
        rows.append(d)
    return pd.DataFrame(rows)

def central_composite(factors, center_points=4, alpha="orthogonal"):
    keys = list(factors.keys())
    base = full_factorial({k: [-1, 1] for k in keys})
    if alpha == "orthogonal":
        a = (len(keys))**0.5
    else:
        a = 1.414213562
    stars = []
    for i, k in enumerate(keys):
        row_pos = {kk: 0 for kk in keys}; row_pos[k] = a; stars.append(row_pos.copy())
        row_neg = {kk: 0 for kk in keys}; row_neg[k] = -a; stars.append(row_neg.copy())
    star_df = pd.DataFrame(stars)
    centers = pd.DataFrame([{kk: 0 for kk in keys} for _ in range(center_points)])
    coded = pd.concat([base.assign(design="factorial"), star_df.assign(design="star"), centers.assign(design="center")], ignore_index=True).reset_index(drop=True)
    def decode(k, x):
        lo, hi = factors[k]
        return ( (x + 1)/2 )*(hi - lo) + lo
    decoded = coded.copy()
    for k in keys:
        decoded[k] = decoded[k].apply(lambda x: decode(k, x))
    decoded.insert(0,"run", range(1, len(decoded)+1))
    return decoded

# =========================
# Main UI
# =========================

# Header
st.markdown('<h1 class="main-header">🧪 R&D Method Development & Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">A comprehensive platform for colorimetric and spectrophotometric assay development</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Analyte selection
    st.markdown("### 📊 Analyte Settings")
    analyte = st.text_input("🏷️ Analyte name", value="Iron", help="Enter the name of your target analyte")
    prof = get_profile(analyte)
    
    st.markdown("### 🔬 Measurement Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        prof["channel"] = st.selectbox(
            "📡 Channel/Wavelength", 
            ["SC_Green","SC_Blue2","SC_Orange","SC_Red","A","Abs","Custom"], 
            index=0,
            help="Select the measurement channel"
        )
    
    with col2:
        prof["absorbance_formula"] = st.selectbox(
            "📐 Formula", 
            ["log10_bg_over_sample","absorbance_single"], 
            index=0,
            format_func=lambda x: "Log₁₀(BG/S)" if x == "log10_bg_over_sample" else "A = -Log₁₀(S/BG)"
        )
    
    if prof["channel"] == "Custom":
        prof["channel"] = st.text_input("Enter custom channel key", value=prof.get("channel_custom","MyChannel"))
        prof["channel_custom"] = prof["channel"]
    
    st.markdown("### 📈 Regression Settings")
    weighting_scheme = st.selectbox(
        "⚖️ Weighting scheme", 
        ["None (OLS)","1/max(C,1)","Variance-weighted (1/SD²)"], 
        index=2,
        help="Choose the weighting scheme for regression"
    )
    
    fit_model = st.selectbox(
        "📉 Calibration model", 
        ["Linear","Quadratic"], 
        index=0
    )
    
    use_rep_means = st.checkbox("📊 Fit using replicate means", value=True)
    expected_reps = st.number_input("🔄 Expected replicates per level", min_value=1, max_value=10, value=2, step=1)
    
    st.markdown("---")
    
    # Profile defaults section
    with st.expander("🧪 Volume Settings", expanded=True):
        prof["defaults"]["base_sample_mL"] = st.number_input(
            "Base sample volume (mL)", 
            1.0, 500.0, 
            float(prof["defaults"].get("base_sample_mL",40.0)), 
            0.5,
            help="The base volume of your sample"
        )
        prof["defaults"]["extra_constant_mL"] = st.number_input(
            "Extra reagent volume (mL)", 
            0.0, 50.0, 
            float(prof["defaults"].get("extra_constant_mL",0.0)), 
            0.1,
            help="Additional constant reagent volume"
        )
        prof["defaults"]["include_all_loc_volumes"] = st.checkbox(
            "Include ALL LOC volumes in final volume", 
            value=bool(prof["defaults"].get("include_all_loc_volumes", True))
        )
    
    # LOC Configuration
    with st.expander("💉 LOC Configuration", expanded=False):
        st.markdown("#### Configure LOC roles and stock concentrations")
        
        # Create a more compact LOC configuration
        for i in range(1,17):
            key = f"LOC{i}"
            row = prof["locs"].get(key, {})
            
            st.markdown(f"**{key}**")
            col1, col2 = st.columns([1,1])
            
            with col1:
                role = st.selectbox(
                    "Role", 
                    ["","standard","reducer","buffer","other"], 
                    index=(["","standard","reducer","buffer","other"].index(row.get("role","")) if row.get("role","") in ["","standard","reducer","buffer","other"] else 0), 
                    key=f"{analyte}_role_{key}",
                    label_visibility="collapsed"
                )
            
            with col2:
                stock = st.number_input(
                    "Stock (mg/L)", 
                    0.0, 1_000_000.0, 
                    float(row.get("stock_mgL",0.0)), 
                    10.0, 
                    key=f"{analyte}_stock_{key}",
                    label_visibility="collapsed"
                )
            
            note = st.text_input(
                "Notes", 
                value=row.get("notes",""), 
                key=f"{analyte}_note_{key}",
                label_visibility="collapsed",
                placeholder="Optional notes..."
            )
            
            if role or stock>0 or note:
                prof["locs"][key] = {"role": role or "", "stock_mgL": stock, "notes": note}
            else:
                if key in prof["locs"]: del prof["locs"][key]
    
    # Profile management
    st.markdown("---")
    st.markdown("### 💾 Profile Management")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "⬇️ Export", 
            data=json.dumps(prof, indent=2), 
            file_name=f"{analyte}_profile.json", 
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        up = st.file_uploader("⬆️ Import", type=["json"], key=f"{analyte}_profile_upload", label_visibility="collapsed")
        if up:
            try:
                p = json.loads(up.getvalue().decode("utf-8"))
                set_profile(analyte, p)
                st.success("✅ Profile loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to load: {e}")
    
    set_profile(analyte, prof)

# Main content area with tabs
tabs = st.tabs(["🔬 Calibration Builder", "🎯 Unknown Prediction", "📋 DOE Designer", "🔍 JSON Explorer", "ℹ️ About"])

# =========================
# Calibration Builder Tab
# =========================
with tabs[0]:
    st.markdown(f'<h2 class="custom-subheader">Calibration Builder — {analyte}</h2>', unsafe_allow_html=True)
    
    # Import section in a collapsible container
    with st.expander("📦 Import Previous Session", expanded=False):
        st.markdown("Load a previously exported project ZIP file to restore your calibration data and models.")
        zup = st.file_uploader("Upload project ZIP", type=["zip"], key="proj_zip_up", label_visibility="collapsed")
        if zup:
            try:
                import zipfile, io, pandas as pd, json
                zf = zipfile.ZipFile(io.BytesIO(zup.getvalue()))
                cand = [n for n in zf.namelist() if n.endswith("_calibration_table.csv")]
                if cand:
                    df_imp = pd.read_csv(zf.open(cand[0]))
                    st.session_state["cal_table"] = df_imp
                    st.success("✅ Calibration table restored from ZIP.")
                candm = [n for n in zf.namelist() if n.endswith("_model.json")]
                if candm:
                    model = json.load(zf.open(candm[0]))
                    st.session_state["imported_model"] = model
                    st.info("📊 Model JSON found and loaded.")
            except Exception as e:
                st.error(f"❌ Import failed: {e}")
    
    # File upload section
    st.markdown("### 📁 Upload Device JSON Files")
    st.info("Upload JSON files containing both Background and Sample scans. The system will automatically detect LOC dosing information for concentration inference.")
    
    files = st.file_uploader(
        "Drop multiple JSON files here", 
        type=["json"], 
        accept_multiple_files=True,
        help="Select one or more JSON files from your device"
    )
    
    rows = []
    if files:
        st.markdown("### 🎯 File Configuration")
        
        for idx, f in enumerate(files):
            with st.expander(f"📄 {f.name}", expanded=True):
                # Create two columns for better organization
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Parse JSON & calculate absorbance
                    try:
                        A, diag, obj = compute_absorbance(f.getvalue(), chan_key=prof["channel"], formula=prof["absorbance_formula"])
                        
                        # Display metrics in a nice card format
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Absorbance", f"{A:.6f}")
                        subcol1, subcol2 = st.columns(2)
                        with subcol1:
                            st.metric("Background", f"{diag['bg_avg']:.2f}", delta=f"n={diag['bg_n']}")
                        with subcol2:
                            st.metric("Sample", f"{diag['sm_avg']:.2f}", delta=f"n={diag['sm_n']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Absorbance calculation failed: {e}")
                        A = None
                        obj = None
                
                with col2:
                    # LOC-based concentration inference
                    conc_calc = 0.0
                    use_auto = True
                    locs = get_loc_doses_from_sample(obj) if obj is not None else {}
                    
                    if locs:
                        st.markdown("**Detected LOC doses (µL):**")
                        loc_display = ", ".join([f"{k}: {v}" for k, v in locs.items()])
                        st.code(loc_display)
                        
                        std_auto, stock_auto = profile_pick_standard(prof, locs)
                        options = ["(no standard)"] + list(locs.keys())
                        idx = 0
                        
                        if std_auto in locs:
                            idx = 1 + list(locs.keys()).index(std_auto)
                            st.success(f"✅ Auto-detected standard: **{std_auto}**")
                        
                        std_loc = st.selectbox(
                            "Select standard LOC", 
                            options, 
                            index=idx, 
                            key=f"std_{f.name}",
                            help="Choose which LOC contains your standard"
                        )
                        
                        # Volume inputs in a compact layout
                        vcol1, vcol2 = st.columns(2)
                        with vcol1:
                            base = st.number_input(
                                "Base volume (mL)", 
                                1.0, 500.0, 
                                float(prof["defaults"]["base_sample_mL"]), 
                                0.5, 
                                key=f"base_{f.name}"
                            )
                        with vcol2:
                            extra = st.number_input(
                                "Extra volume (mL)", 
                                0.0, 50.0, 
                                float(prof["defaults"]["extra_constant_mL"]), 
                                0.1, 
                                key=f"extra_{f.name}"
                            )
                        
                        include_all = st.checkbox(
                            "Include ALL LOC volumes", 
                            value=bool(prof["defaults"]["include_all_loc_volumes"]), 
                            key=f"incl_{f.name}"
                        )
                        
                        total_loc_mL = sum(locs.values())/1000.0 if include_all else 0.0
                        
                        if std_loc == "(no standard)":
                            conc_calc = 0.0
                            st.info("ℹ️ No standard selected → **0.0000 mg/L** (blank)")
                        else:
                            default_stock = stock_auto if (std_auto==std_loc and stock_auto) else float(prof["locs"].get(std_loc,{}).get("stock_mgL",0.0) or 1000.0)
                            stock = st.number_input(
                                f"{std_loc} stock (mg/L)", 
                                0.0, 1_000_000.0, 
                                float(default_stock), 
                                10.0, 
                                key=f"stock_{f.name}"
                            )
                            conc_calc = compute_spike_conc_mgL(stock, float(locs.get(std_loc,0.0)), base_sample_mL=base, extra_mL=extra+total_loc_mL)
                            
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown(f"**Calculated concentration: {conc_calc:.4f} mg/L**")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            use_auto = st.checkbox("Use calculated concentration", value=True, key=f"useauto_{f.name}")
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("⚠️ No LOC doses detected → defaulting to blank (0 mg/L)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    manual = st.number_input(
                        "Manual concentration (mg/L)", 
                        0.0, 1e9, 0.0, 0.1, 
                        key=f"man_{f.name}",
                        help="Override with manual concentration if > 0"
                    )
                    
                    final_c = manual if manual > 0 else (conc_calc if use_auto else 0.0)
                    
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown(f"**Final concentration: {final_c:.4f} mg/L**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                rows.append({
                    "file": f.name, 
                    "analyte": analyte, 
                    "channel": prof["channel"], 
                    "A": A, 
                    "conc_mgL": final_c
                })
        
        # Add to calibration table button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("➕ Add to Calibration Table", type="primary", use_container_width=True):
                df = pd.DataFrame(rows)
                st.session_state["cal_table"] = df
                st.success("✅ Files added to calibration table!")
    
    # Display calibration table if exists
    if "cal_table" in st.session_state:
        df = st.session_state["cal_table"]
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Calibration Data")
        
        # Display the dataframe with custom styling
        st.dataframe(
            df.style.format({
                'A': '{:.6f}',
                'conc_mgL': '{:.4f}'
            }),
            use_container_width=True
        )
        
        # Aggregate function
        def aggregate(df, analyte):
            sub = df[df["analyte"]==analyte].dropna(subset=["A","conc_mgL"]).copy()
            if sub.empty: return None, None
            grp = sub.groupby("conc_mgL", as_index=False).agg(
                n=("A","count"), 
                A_mean=("A","mean"), 
                A_sd=("A","std")
            )
            grp["A_sd"].fillna(0.0, inplace=True)
            grp["A_rsd_%"] = np.where(grp["A_mean"]!=0, grp["A_sd"]/grp["A_mean"]*100.0, np.nan)
            grp["meets_n"] = grp["n"] >= expected_reps
            return sub, grp
        
        sub, grp = aggregate(df, analyte)
        
        if grp is not None:
            st.markdown("### 📈 Replicate Statistics")
            
            # Display replicate summary with conditional formatting
            st.dataframe(
                grp.style.format({
                    'conc_mgL': '{:.4f}',
                    'A_mean': '{:.6f}',
                    'A_sd': '{:.6f}',
                    'A_rsd_%': '{:.2f}'
                }).apply(lambda x: ['background-color: #d4edda' if v else 'background-color: #f8d7da' 
                                   for v in x], subset=['meets_n'], axis=0),
                use_container_width=True
            )
        
        # Model fitting function
        def fit(sub, grp, title):
            if sub is None and grp is None: return None, None, None
            use_means = use_rep_means and (grp is not None) and (not grp.empty)
            if use_means:
                xs = grp["conc_mgL"].astype(float).tolist()
                ys = grp["A_mean"].astype(float).tolist()
                sds = grp["A_sd"].astype(float).tolist()
                blanks = grp[grp["conc_mgL"]==0]["A_mean"].astype(float).tolist()
            else:
                xs = sub["conc_mgL"].astype(float).tolist()
                ys = sub["A"].astype(float).tolist()
                sds = None
                blanks = sub[sub["conc_mgL"]==0]["A"].astype(float).tolist()
            
            xy = [(x,y,(sds[i] if sds is not None and i<len(sds) else None)) 
                  for i,(x,y) in enumerate(zip(xs,ys)) 
                  if np.isfinite(x) and np.isfinite(y)]
            
            if len(xy)<2:
                st.warning("⚠️ Need at least 2 points to fit a model.")
                return None, None, None
            
            xs=[x for x,_,_ in xy]
            ys=[y for _,y,_ in xy]
            sds=[sd for _,_,sd in xy] if use_means else None
            
            if len(set([round(x,6) for x in xs]))<2:
                st.warning("⚠️ Need at least 2 unique concentration levels.")
                return None,None,None
            
            weights=None
            if weighting_scheme=="1/max(C,1)":
                weights=[1.0/max(x,1.0) for x in xs]
            elif weighting_scheme=="Variance-weighted (1/SD²)" and use_means:
                eps=1e-6
                nz=[sd for sd in sds if sd and sd>0]
                base=np.median(nz) if nz else 0.01
                denom=[(sd if (sd and sd>0) else base)**2 + eps for sd in sds]
                weights=[1.0/d for d in denom]
            
            if fit_model=="Quadratic":
                try:
                    a,b,c,R2 = fit_quadratic(xs, ys, weights=weights)
                    png,pdf = plot_calibration(xs,ys,quad=(a,b,c),title=title)
                    res = {"model":"quadratic","a":a,"b":b,"c":c,"R2":R2}
                    lod=loq=None
                    sd_blank=None
                except Exception as e:
                    st.error(f"❌ Quadratic fit failed: {e}")
                    return None,None,None
            else:
                try:
                    m,b,R2 = fit_linear(xs, ys, weights=weights)
                    png,pdf = plot_calibration(xs,ys,m=m,b=b,title=title)
                    lod,loq,sd_blank = lod_loq(blanks, m)
                    res = {"model":"linear","m":m,"b":b,"R2":R2,"LoD":lod,"LoQ":loq,"blank_sd_A":sd_blank}
                except Exception as e:
                    st.error(f"❌ Linear fit failed: {e}")
                    return None,None,None
            
            res.update({
                "n_points": len(xs),
                "levels": sorted(set(xs)),
                "weighting": weighting_scheme,
                "used_replicate_means": use_means
            })
            
            return (res, png, pdf, xs, ys)
        
        # Perform model fitting
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Model Fitting Results")
        
        out = fit(sub, grp, f"{analyte} Calibration")
        if out:
            fit_res, png, pdf, xs_fit, ys_fit = out
            
            # Display fit results in organized columns
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("#### 📊 Model Parameters")
                if fit_res["model"] == "linear":
                    st.metric("Slope (m)", f"{fit_res['m']:.6f}")
                    st.metric("Intercept (b)", f"{fit_res['b']:.6f}")
                else:
                    st.metric("a", f"{fit_res['a']:.6f}")
                    st.metric("b", f"{fit_res['b']:.6f}")
                    st.metric("c", f"{fit_res['c']:.6f}")
            
            with col2:
                st.markdown("#### 📈 Model Quality")
                st.metric("R²", f"{fit_res['R2']:.4f}")
                st.metric("Points", fit_res['n_points'])
                st.metric("Weighting", fit_res['weighting'])
            
            with col3:
                st.markdown("#### 🎯 Detection Limits")
                if fit_res.get("LoD") is not None:
                    st.metric("LoD (mg/L)", f"{fit_res.get('LoD', 0):.4f}")
                    st.metric("LoQ (mg/L)", f"{fit_res.get('LoQ', 0):.4f}")
                else:
                    st.info("N/A for quadratic models")
            
            # Display calibration plot
            if png:
                st.image(png, caption=f"{analyte} Calibration Curve", use_column_width=True)
            
            # Advanced diagnostics section
            with st.expander("🔬 Advanced Diagnostics", expanded=False):
                # Residual analysis
                if fit_res["model"]=="linear":
                    model_for_resid = {"model":"linear","m":fit_res["m"],"b":fit_res["b"]}
                    loocv = loocv_metrics(xs_fit, ys_fit, model_type="linear")
                else:
                    model_for_resid = {"model":"quadratic","a":fit_res["a"],"b":fit_res["b"],"c":fit_res["c"]}
                    loocv = loocv_metrics(xs_fit, ys_fit, model_type="quadratic")
                
                png_res, pdf_res = plot_residuals(xs_fit, ys_fit, model_for_resid, title=f"{analyte} Residuals")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### LOOCV Results")
                    st.json(loocv)
                
                with col2:
                    st.markdown("#### Residual Plot")
                    st.image(png_res, use_column_width=True)
                
                # Full model JSON
                st.markdown("#### Complete Model JSON")
                model = {
                    "analyte": analyte,
                    "created_at": datetime.utcnow().isoformat()+"Z",
                    "channel": prof["channel"],
                    "absorbance_formula": prof["absorbance_formula"],
                    "fit": fit_res,
                    "range_hint_mgL": [0.0, 25.0]
                }
                st.json(model)
            
            # Download section
            st.markdown("### 💾 Export Options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.download_button(
                    "📊 Model JSON",
                    data=json.dumps(model, indent=2),
                    file_name=f"{analyte}_model.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    "📈 Plot (PNG)",
                    data=png,
                    file_name=f"{analyte}_calibration.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col3:
                st.download_button(
                    "📄 Plot (PDF)",
                    data=pdf,
                    file_name=f"{analyte}_calibration.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col4:
                # Create session ZIP
                plots = {}
                if png: plots[f"{analyte}_calibration.png"] = png
                if png_res: plots[f"{analyte}_residuals.png"] = png_res
                zip_bytes = export_session_zip(analyte, prof, cal_table=df, model=model, plots=plots)
                
                st.download_button(
                    "📦 Session ZIP",
                    data=zip_bytes,
                    file_name=f"{analyte}_session.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )

# =========================
# Unknown Prediction Tab
# =========================
with tabs[1]:
    st.markdown('<h2 class="custom-subheader">Unknown Sample Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Load Model")
        model_file = st.file_uploader(
            "Upload model JSON", 
            type=["json"], 
            key="model_up",
            help="Upload a model JSON file created in the Calibration Builder"
        )
    
    with col2:
        st.markdown("### 📁 Sample Files")
        runs = st.file_uploader(
            "Upload sample JSON files", 
            type=["json"], 
            accept_multiple_files=True, 
            key="run_up",
            help="Upload one or more unknown sample files"
        )
    
    if model_file and runs:
        try:
            model = json.loads(model_file.getvalue().decode("utf-8"))
        except Exception:
            model = json.loads(model_file.getvalue())
        
        # Display model info
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**Model**: {model.get('analyte', 'Unknown')} | **Type**: {model['fit']['model']} | **R²**: {model['fit'].get('R2', 'N/A'):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        chan = model.get("channel","SC_Green")
        formula = model.get("absorbance_formula","log10_bg_over_sample")
        
        # Process samples
        results = []
        for f in runs:
            A, diag, _ = compute_absorbance(f.getvalue(), chan_key=chan, formula=formula)
            
            if model["fit"]["model"]=="linear":
                m=model["fit"]["m"]
                b=model["fit"]["b"]
                C = predict_conc_linear(A, m, b)
            else:
                a=model["fit"]["a"]
                bq=model["fit"]["b"]
                c=model["fit"]["c"]
                A0 = c - A
                disc = bq*bq - 4*a*A0
                if disc < 0:
                    C = float("nan")
                else:
                    r1 = (-bq + math.sqrt(disc))/(2*a) if a!=0 else float("nan")
                    r2 = (-bq - math.sqrt(disc))/(2*a) if a!=0 else float("nan")
                    lo,hi = model.get("range_hint_mgL",[0,25])
                    mid=(lo+hi)/2.0
                    candidates = [r for r in [r1,r2] if np.isfinite(r)]
                    C = min(candidates, key=lambda r: abs(r-mid)) if candidates else float("nan")
            
            results.append({
                "File": f.name,
                "Absorbance": f"{A:.6f}",
                "Concentration (mg/L)": f"{C:.4f}",
                "Background": f"{diag['bg_avg']:.2f}",
                "Sample": f"{diag['sm_avg']:.2f}"
            })
        
        # Display results
        st.markdown("### 📊 Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            "💾 Download Results (CSV)",
            data=csv,
            file_name=f"{model.get('analyte', 'unknown')}_predictions.csv",
            mime="text/csv"
        )

# =========================
# DOE Designer Tab
# =========================
with tabs[2]:
    st.markdown('<h2 class="custom-subheader">Design of Experiments (DOE)</h2>', unsafe_allow_html=True)
    
    design = st.selectbox(
        "🔬 Select experiment design",
        ["Calibration (auto-generator)","Calibration (manual levels)","2-level full factorial (screening)","Central Composite (CCD)"],
        index=0,
        help="Choose the type of experimental design you want to create"
    )
    
    reps = st.number_input("🔄 Replicates per condition", 1, 10, 2, 1)
    
    templ = DOE_TEMPLATES.get(analyte, DOE_TEMPLATES.get("Iron"))
    
    if design == "Calibration (auto-generator)":
        st.markdown("### 🎯 Automatic Calibration Level Generator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cmin = st.number_input("Min concentration (mg/L)", 0.0, 1e9, 0.0, 0.01)
        
        with col2:
            cmax = st.number_input("Max concentration (mg/L)", 0.0, 1e9, 25.0, 0.01)
        
        with col3:
            npts = st.number_input("Number of non-zero points", 2, 15, 6, 1)
        
        scheme = st.selectbox(
            "Spacing scheme",
            ["Linear", "Logarithmic", "ICH template"],
            index=0,
            help="Choose how calibration points are distributed"
        )
        
        include_blank = st.checkbox("Include blank (0 mg/L)", value=True)
        
        if st.button("🎨 Generate Calibration Plan", type="primary"):
            levels = []
            if scheme == "Linear":
                levels = list(np.linspace(cmin, cmax, npts))
            elif scheme == "Logarithmic":
                low = max(cmin, 1e-6)
                levels = list(np.logspace(np.log10(low), np.log10(max(cmax, low*10)), npts))
            else:  # ICH template
                lo, hi = min(cmin, cmax), max(cmin, cmax)
                perc = [0.2, 0.4, 0.6, 0.8, 1.0]
                levels = [lo + p*(hi-lo) for p in perc]
            
            levels = [round(x, 6) for x in levels if x >= 0]
            seq = ([] if not include_blank else [0.0]) + levels
            seq = seq * reps
            random.shuffle(seq)
            
            rows = [{"order": i+1, "analyte": analyte, "target_mgL": c, "json_path": "", "notes": ""} 
                   for i, c in enumerate(seq)]
            plan = pd.DataFrame(rows)
            
            st.markdown("### 📋 Generated Calibration Plan")
            st.dataframe(plan, use_container_width=True)
            
            csv = plan.to_csv(index=False)
            st.download_button(
                "💾 Download Calibration Plan (CSV)",
                data=csv,
                file_name=f"{analyte}_calibration_auto_plan.csv",
                mime="text/csv"
            )
    
    elif design == "Calibration (manual levels)":
        st.markdown("### 🎯 Manual Calibration Levels")
        
        levels_str = st.text_input(
            "Enter concentration levels (mg/L)",
            value="0, 1, 2, 5, 10, 15, 20, 25",
            help="Enter comma-separated concentration values"
        )
        
        if st.button("🎨 Generate Manual Plan", type="primary"):
            try:
                levels = [float(x.strip()) for x in levels_str.split(",") if x.strip()!=""]
            except Exception:
                st.error("❌ Could not parse levels. Please use comma-separated numbers.")
                levels = [0,1,2,5,10,15,20,25]
            
            seq = levels * reps
            random.shuffle(seq)
            
            rows = [{"order": i+1, "analyte": analyte, "target_mgL": c, "json_path": "", "notes": ""} 
                   for i,c in enumerate(seq)]
            plan = pd.DataFrame(rows)
            
            st.markdown("### 📋 Generated Calibration Plan")
            st.dataframe(plan, use_container_width=True)
            
            csv = plan.to_csv(index=False)
            st.download_button(
                "💾 Download Manual Calibration Plan (CSV)",
                data=csv,
                file_name=f"{analyte}_calibration_plan.csv",
                mime="text/csv"
            )
    
    elif design == "2-level full factorial (screening)":
        st.markdown("### 🔬 2-Level Full Factorial Design")
        st.info("Enter factors with their low and high values for screening experiments.")
        
        # Preset factors for the analyte
        preset_txt = templ["screening"] if templ else "pH: 3, 6\nBuffer_mM: 10, 50\nReagent1_uL: 100, 1000\nReactionTime_min: 1, 10\nTemperature_C: 20, 35"
        
        raw = st.text_area(
            "Factor definitions",
            value=preset_txt,
            height=150,
            help="Format: FactorName: low_value, high_value"
        )
        
        if st.button("🎯 Generate Factorial Design", type="primary"):
            factors = {}
            for line in raw.splitlines():
                if ":" in line and "," in line:
                    k, rest = line.split(":",1)
                    lo, hi = rest.split(",",1)
                    try:
                        factors[k.strip()] = [float(lo), float(hi)]
                    except Exception:
                        pass
            
            if not factors:
                st.error("❌ No valid factors parsed. Please check your format.")
            else:
                df = full_factorial(factors)
                rows = []
                for _ in range(reps):
                    for _, r in df.iterrows():
                        row = {"analyte": analyte}
                        row.update({k: r[k] for k in df.columns if k!="run"})
                        rows.append(row)
                
                random.shuffle(rows)
                plan = pd.DataFrame(rows)
                plan.insert(0,"order", range(1,len(rows)+1))
                
                st.markdown("### 📊 Generated Factorial Design")
                st.info(f"**Total runs**: {len(plan)} ({len(factors)} factors, {2**len(factors)} combinations × {reps} replicates)")
                st.dataframe(plan, use_container_width=True)
                
                csv = plan.to_csv(index=False)
                st.download_button(
                    "💾 Download Factorial Design (CSV)",
                    data=csv,
                    file_name=f"{analyte}_factorial_plan.csv",
                    mime="text/csv"
                )
    
    else:  # CCD
        st.markdown("### 🎯 Central Composite Design (CCD)")
        st.info("Enter factors with their low and high values for response surface methodology.")
        
        preset_txt = templ["ccd"] if templ else "pH: 3, 6\nBuffer_mM: 10, 50\nReagent1_uL: 100, 1000\nReactionTime_min: 1, 10"
        
        raw = st.text_area(
            "Factor definitions",
            value=preset_txt,
            height=150,
            help="Format: FactorName: low_value, high_value"
        )
        
        center_pts = st.number_input("Center points", 0, 20, 4, 1)
        
        if st.button("🎯 Generate CCD", type="primary"):
            factors = {}
            for line in raw.splitlines():
                if ":" in line and "," in line:
                    k, rest = line.split(":",1)
                    lo, hi = rest.split(",",1)
                    try:
                        factors[k.strip()] = (float(lo), float(hi))
                    except Exception:
                        pass
            
            if not factors:
                st.error("❌ No valid factors parsed. Please check your format.")
            else:
                df = central_composite(factors, center_points=center_pts, alpha="orthogonal")
                rows = []
                for _ in range(reps):
                    for _, r in df.iterrows():
                        row = {"analyte": analyte}
                        for k in factors.keys():
                            row[k] = r[k]
                        row["design"] = r.get("design","")
                        rows.append(row)
                
                random.shuffle(rows)
                plan = pd.DataFrame(rows)
                plan.insert(0,"order", range(1,len(rows)+1))
                
                st.markdown("### 📊 Generated CCD Design")
                
                # Show design summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Factorial points", len(df[df['design']=='factorial']))
                with col2:
                    st.metric("Star points", len(df[df['design']=='star']))
                with col3:
                    st.metric("Center points", len(df[df['design']=='center']))
                
                st.dataframe(plan, use_container_width=True)
                
                csv = plan.to_csv(index=False)
                st.download_button(
                    "💾 Download CCD Design (CSV)",
                    data=csv,
                    file_name=f"{analyte}_CCD_plan.csv",
                    mime="text/csv"
                )

# =========================
# JSON Explorer Tab
# =========================
with tabs[3]:
    st.markdown('<h2 class="custom-subheader">JSON File Explorer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        f = st.file_uploader("Upload a device JSON file", type=["json"], key="explore")
    
    with col2:
        chan = st.text_input("Channel key", value=prof["channel"], key="explore_chan")
        formula = st.selectbox(
            "Absorbance formula",
            ["log10_bg_over_sample","absorbance_single"],
            index=["log10_bg_over_sample","absorbance_single"].index(prof["absorbance_formula"]) if prof.get("absorbance_formula") in ["log10_bg_over_sample","absorbance_single"] else 0,
            key="explore_formula"
        )
    
    if f:
        try:
            A, diag, obj = compute_absorbance(f.getvalue(), chan_key=chan, formula=formula)
            
            st.markdown("### 📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Absorbance", f"{A:.6f}")
            
            with col2:
                st.metric("Background Avg", f"{diag['bg_avg']:.2f}", delta=f"n={diag['bg_n']}")
            
            with col3:
                st.metric("Sample Avg", f"{diag['sm_avg']:.2f}", delta=f"n={diag['sm_n']}")
            
            locs = get_loc_doses_from_sample(obj)
            if locs:
                st.markdown("### 💉 Detected LOC Doses")
                loc_df = pd.DataFrame([{"LOC": k, "Volume (µL)": v} for k, v in locs.items()])
                st.dataframe(loc_df, use_container_width=True)
            
            # JSON structure viewer
            with st.expander("📄 View Raw JSON Structure", expanded=False):
                st.json(obj)
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# =========================
# About Tab
# =========================
with tabs[4]:
    st.markdown('<h2 class="custom-subheader">About R&D MD and Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Overview
        
        A comprehensive platform for **method development** and **analysis** of colorimetric and spectrophotometric assays.
        This application streamlines the entire workflow from calibration to sample analysis.
        
        ### 🌟 Key Features
        
        - **🔬 Multi-Analyte Support**: User-defined analytes with customizable channels and formulas
        - **📊 Intelligent Calibration**: Automatic LOC-based spike inference with per-analyte profiles
        - **📈 Advanced Statistics**: Replicate-aware calibration with multiple weighting schemes
        - **🎯 Flexible Models**: Linear and quadratic calibration models with quality metrics
        - **📋 DOE Generator**: Create calibration sets, factorial designs, and CCD experiments
        - **💾 Complete Traceability**: Session export/import for collaboration and documentation
        
        ### 📐 Technical Capabilities
        
        - **Robust JSON parsing** with nested scan support
        - **Variance-weighted regression** (1/SD²) for improved accuracy
        - **LOD/LOQ calculations** following analytical guidelines
        - **LOOCV diagnostics** for model validation
        - **Residual analysis** for quality assessment
        - **PNG/PDF export** for publication-quality figures
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Important Notes
        
        <div class="warning-box">
        <strong>Analyte Stability Considerations</strong><br><br>
        For redox-active species (e.g., Fe²⁺/Fe³⁺), speciation can change rapidly after sampling. 
        For accurate speciation analysis:<br>
        • Preserve samples immediately<br>
        • Minimize holding time<br>
        • Use appropriate complexation<br>
        • Note that acid preservation stabilizes total analyte but not speciation
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Session management section
    st.markdown("### 💾 Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📦 Import Session
        
        Load a previously exported session to restore:
        - Analyte profiles and settings
        - Calibration data
        - Fitted models
        - Generated plots
        """)
        
        zip_up = st.file_uploader("Upload session ZIP file", type=["zip"], key="sess_zip")
        
        if zip_up:
            try:
                loaded = import_session_zip(zip_up.getvalue())
                
                success_items = []
                
                if loaded.get("profile"):
                    set_profile(analyte, loaded["profile"])
                    success_items.append("✅ Profile settings")
                
                if loaded.get("cal_table") is not None:
                    st.session_state["cal_table"] = loaded["cal_table"]
                    success_items.append("✅ Calibration table")
                
                if loaded.get("model") is not None:
                    st.session_state["loaded_model"] = loaded["model"]
                    success_items.append("✅ Model parameters")
                
                if success_items:
                    st.success("Session loaded successfully!")
                    for item in success_items:
                        st.write(item)
                    st.info("📍 Navigate to other tabs to view imported data")
                
            except Exception as e:
                st.error(f"❌ Failed to import session: {e}")
    
    with col2:
        st.markdown("""
        #### 📤 Export Current Session
        
        The session ZIP includes:
        - Complete analyte profile configuration
        - Calibration data table
        - Fitted model parameters
        - All generated plots
        - Session metadata
        
        **Pro tip**: Export sessions regularly for backup and to share with collaborators!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>R&D MD and Analysis v2.0 | Built with Streamlit</p>
        <p>For support and documentation, contact your system administrator</p>
    </div>
    """, unsafe_allow_html=True)
