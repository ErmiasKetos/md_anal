#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import json, math, io, random, zipfile
import statistics as stats
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="R&D MD and Analysis", layout="wide")

# =========================
# Generic utilities
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
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(xs, ys)
    if len(xs)>=2:
        x_min, x_max = min(xs), max(xs); span = x_max - x_min
        grid_x = np.linspace(x_min-0.05*span, x_max+0.05*span if span>0 else x_max+1, 200)
        if quad is not None:
            a,b2,c = quad
            ax.plot(grid_x, a*grid_x**2 + b2*grid_x + c)
        elif m is not None and b is not None:
            ax.plot(grid_x, m*grid_x + b)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout()
    png = io.BytesIO(); fig.savefig(png, format="png", dpi=200, bbox_inches="tight"); png.seek(0)
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
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(yhat, resid)
    ax.axhline(0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Fitted y")
    ax.set_ylabel("Residuals")
    fig.tight_layout()
    png = io.BytesIO(); fig.savefig(png, format="png", dpi=200, bbox_inches="tight"); png.seek(0)
    pdf = io.BytesIO(); fig.savefig(pdf, format="pdf", bbox_inches="tight"); pdf.seek(0)
    plt.close(fig); return png.getvalue(), pdf.getvalue()

def export_session_zip(analyte, profile, cal_table=None, model=None, plots=None):
    # Bundle current session into a ZIP: profile.json, calibration.csv, model.json, plots/*.png
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


# =========================
# Fitting helpers: residuals, LOOCV, and zip export
# =========================

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
    # RMSE and MAE on finite errors
    fe = np.array([e for e in errs if np.isfinite(e)])
    rmse = float(np.sqrt(np.mean(fe**2))) if fe.size else float("nan")
    mae = float(np.mean(np.abs(fe))) if fe.size else float("nan")
    return {"errors": errs, "RMSE": rmse, "MAE": mae}

def make_residual_plot(xs, res, title="Residuals vs Conc", xlabel="Conc (mg/L)"):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(xs, res)
    ax.axhline(0, linestyle="--")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Residual (A)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight"); buf.seek(0); plt.close(fig)
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
# =========================
# Profiles (per-analyte)
# =========================

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

# =========================
# Sidebar configuration
# =========================

st.title("R&D MD and Analysis")

with st.sidebar:
    st.header("Global Settings")
    analyte = st.text_input("Analyte name", value="Iron")
    prof = get_profile(analyte)

    prof["channel"] = st.selectbox("Channel / Wavelength key", ["SC_Green","SC_Blue2","SC_Orange","SC_Red","A","Abs","Custom"], index=0)
    if prof["channel"] == "Custom":
        prof["channel"] = st.text_input("Enter custom channel key", value=prof.get("channel_custom","MyChannel"))
        prof["channel_custom"] = prof["channel"]

    prof["absorbance_formula"] = st.selectbox("Absorbance formula", ["log10_bg_over_sample","absorbance_single"], index=0)

    weighting_scheme = st.selectbox("Regression weighting", ["None (OLS)","1/max(C,1)","Variance-weighted (1/SD^2)"], index=2)
    fit_model = st.selectbox("Calibration model", ["Linear","Quadratic"], index=0)
    use_rep_means = st.checkbox("Fit using replicate means", value=True)
    expected_reps = st.number_input("Expected replicates per level", min_value=1, max_value=10, value=2, step=1)

    st.markdown("---")
    st.markdown("### Profile Defaults")
    prof["defaults"]["base_sample_mL"] = st.number_input("Base sample volume (mL)", 1.0, 500.0, float(prof["defaults"].get("base_sample_mL",40.0)), 0.5)
    prof["defaults"]["extra_constant_mL"] = st.number_input("Extra constant reagent volume (mL)", 0.0, 50.0, float(prof["defaults"].get("extra_constant_mL",0.0)), 0.1)
    prof["defaults"]["include_all_loc_volumes"] = st.checkbox("Include ALL LOC volumes in final volume by default", value=bool(prof["defaults"].get("include_all_loc_volumes", True)))

    st.markdown("### LOC Map (roles & stock)")
    for i in range(1,17):
        key = f"LOC{i}"
        row = prof["locs"].get(key, {})
        cols = st.columns([1,1,1,2])
        with cols[0]:
            role = st.selectbox(f"{key} role", ["","standard","reducer","buffer","other"], index=(["","standard","reducer","buffer","other"].index(row.get("role","")) if row.get("role","") in ["","standard","reducer","buffer","other"] else 0), key=f"{analyte}_role_{key}")
        with cols[1]:
            stock = st.number_input(f"{key} stock (mg/L)", 0.0, 1_000_000.0, float(row.get("stock_mgL",0.0)), 10.0, key=f"{analyte}_stock_{key}")
        with cols[2]:
            note = st.text_input(f"{key} notes", value=row.get("notes",""), key=f"{analyte}_note_{key}")
        if role or stock>0 or note:
            prof["locs"][key] = {"role": role or "", "stock_mgL": stock, "notes": note}
        else:
            if key in prof["locs"]: del prof["locs"][key]

    colp1,colp2 = st.columns(2)
    with colp1:
        st.download_button("Download profile JSON", data=json.dumps(prof, indent=2), file_name=f"{analyte}_profile.json", mime="application/json")
    with colp2:
        up = st.file_uploader("Load profile JSON", type=["json"], key=f"{analyte}_profile_upload")
        if up:
            try:
                p = json.loads(up.getvalue().decode("utf-8"))
                set_profile(analyte, p); st.success("Profile loaded."); st.rerun()
            except Exception as e:
                st.error(f"Failed to load profile: {e}")

    set_profile(analyte, prof)

tabs = st.tabs(["Calibration Builder","Unknown Prediction","DOE Designer","JSON Explorer","About"])

# =========================
# Calibration Builder
# =========================

with tabs[0]:
    st.subheader(f"Calibration Builder — {analyte}")
    with st.expander("Import Project ZIP (optional)", expanded=False):
        zup = st.file_uploader("Upload a project ZIP exported by this app", type=["zip"], key="proj_zip_up")
        if zup:
            try:
                import zipfile, io, pandas as pd, json
                zf = zipfile.ZipFile(io.BytesIO(zup.getvalue()))
                # Try to read calibration table
                cand = [n for n in zf.namelist() if n.endswith("_calibration_table.csv")]
                if cand:
                    df_imp = pd.read_csv(zf.open(cand[0]))
                    st.session_state["cal_table"] = df_imp
                    st.success("Calibration table restored from ZIP.")
                # Try to read model
                candm = [n for n in zf.namelist() if n.endswith("_model.json")]
                if candm:
                    model = json.load(zf.open(candm[0]))
                    st.session_state["imported_model"] = model
                    st.info("Model JSON found in ZIP (available under Unknown Prediction if you download & reuse).")
            except Exception as e:
                st.error(f"Failed to import project ZIP: {e}")
    st.subheader(f"Calibration Builder — {analyte}")
    st.write("Upload device JSON files (Background + Sample in the same file). Optionally infer concentration from LOC dosing.")
    files = st.file_uploader("Drop multiple JSON files", type=["json"], accept_multiple_files=True)

    rows = []
    if files:
        st.markdown("### Assign concentration & channel to each file")
        for f in files:
            with st.expander(f"File: {f.name}", expanded=True):
                # parse JSON & absorbance
                try:
                    A, diag, obj = compute_absorbance(f.getvalue(), chan_key=prof["channel"], formula=prof["absorbance_formula"])
                    st.code(f"A={A:.6f} | BG={diag['bg_avg']:.2f} | S={diag['sm_avg']:.2f}")
                except Exception as e:
                    st.error(f"Absorbance failed: {e}"); A=None; obj=None

                # LOC-based inference
                conc_calc = 0.0; use_auto = True
                locs = get_loc_doses_from_sample(obj) if obj is not None else {}
                if locs:
                    st.write("Detected LOC doses (µL):", locs)
                    std_auto, stock_auto = profile_pick_standard(prof, locs)
                    options = ["(no standard)"] + list(locs.keys())
                    idx = 0
                    if std_auto in locs: idx = 1 + list(locs.keys()).index(std_auto); st.success(f"Profile suggests standard at **{std_auto}**")
                    std_loc = st.selectbox("Select standard LOC", options, index=idx, key=f"std_{f.name}")
                    base = st.number_input("Base sample volume (mL)", 1.0, 500.0, float(prof["defaults"]["base_sample_mL"]), 0.5, key=f"base_{f.name}")
                    extra = st.number_input("Extra constant reagent volume (mL)", 0.0, 50.0, float(prof["defaults"]["extra_constant_mL"]), 0.1, key=f"extra_{f.name}")
                    include_all = st.checkbox("Include ALL LOC volumes in final volume", value=bool(prof["defaults"]["include_all_loc_volumes"]), key=f"incl_{f.name}")
                    total_loc_mL = sum(locs.values())/1000.0 if include_all else 0.0
                    if std_loc == "(no standard)":
                        conc_calc = 0.0
                        st.info("No standard selected → **0.0000 mg/L** (blank).")
                    else:
                        default_stock = stock_auto if (std_auto==std_loc and stock_auto) else float(prof["locs"].get(std_loc,{}).get("stock_mgL",0.0) or 1000.0)
                        stock = st.number_input(f"{std_loc} stock (mg/L)", 0.0, 1_000_000.0, float(default_stock), 10.0, key=f"stock_{f.name}")
                        conc_calc = compute_spike_conc_mgL(stock, float(locs.get(std_loc,0.0)), base_sample_mL=base, extra_mL=extra+total_loc_mL)
                        st.info(f"Calculated conc: **{conc_calc:.4f} mg/L**")
                        use_auto = st.checkbox("Use calculated concentration", value=True, key=f"useauto_{f.name}")
                else:
                    st.warning("No LOC doses found → defaulting to blank unless you provide a manual concentration.")

                manual = st.number_input("Manual concentration (mg/L) (overrides if >0)", 0.0, 1e9, 0.0, 0.1, key=f"man_{f.name}")
                final_c = manual if manual>0 else (conc_calc if use_auto else 0.0)
                st.caption(f"Final concentration used: {final_c:.4f} mg/L")

                rows.append({"file": f.name, "analyte": analyte, "channel": prof["channel"], "A": A, "conc_mgL": final_c})

        if st.button("Add to calibration table"):
            df = pd.DataFrame(rows)
            st.session_state["cal_table"] = df

    if "cal_table" in st.session_state:
        df = st.session_state["cal_table"]
        st.markdown("### Calibration table")
        st.dataframe(df)

        def aggregate(df, analyte):
            sub = df[df["analyte"]==analyte].dropna(subset=["A","conc_mgL"]).copy()
            if sub.empty: return None, None
            grp = sub.groupby("conc_mgL", as_index=False).agg(n=("A","count"), A_mean=("A","mean"), A_sd=("A","std"))
            grp["A_sd"].fillna(0.0, inplace=True)
            grp["A_rsd_%"] = np.where(grp["A_mean"]!=0, grp["A_sd"]/grp["A_mean"]*100.0, np.nan)
            grp["meets_n"] = grp["n"] >= expected_reps
            return sub, grp

        sub, grp = aggregate(df, analyte)
        st.markdown("### Replicate summary")
        if grp is not None: st.dataframe(grp)

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
            xy = [(x,y,(sds[i] if sds is not None and i<len(sds) else None)) for i,(x,y) in enumerate(zip(xs,ys)) if np.isfinite(x) and np.isfinite(y)]
            if len(xy)<2: st.warning("Need ≥2 points to fit."); return None, None, None
            xs=[x for x,_,_ in xy]; ys=[y for _,y,_ in xy]; sds=[sd for _,_,sd in xy] if use_means else None
            if len(set([round(x,6) for x in xs]))<2: st.warning("Need ≥2 unique levels."); return None,None,None
            weights=None
            if weighting_scheme=="1/max(C,1)":
                weights=[1.0/max(x,1.0) for x in xs]
            elif weighting_scheme=="Variance-weighted (1/SD^2)" and use_means:
                eps=1e-6; nz=[sd for sd in sds if sd and sd>0]; base=np.median(nz) if nz else 0.01
                denom=[(sd if (sd and sd>0) else base)**2 + eps for sd in sds]; weights=[1.0/d for d in denom]
            if fit_model=="Quadratic":
                try:
                    a,b,c,R2 = fit_quadratic(xs, ys, weights=weights)
                    png,pdf = plot_calibration(xs,ys,quad=(a,b,c),title=title)
                    res = {"model":"quadratic","a":a,"b":b,"c":c,"R2":R2}
                    lod=loq=None; sd_blank=None
                except Exception as e:
                    st.error(f"Quadratic fit failed: {e}"); return None,None,None
            else:
                try:
                    m,b,R2 = fit_linear(xs, ys, weights=weights)
                    png,pdf = plot_calibration(xs,ys,m=m,b=b,title=title)
                    lod,loq,sd_blank = lod_loq(blanks, m)
                    res = {"model":"linear","m":m,"b":b,"R2":R2,"LoD":lod,"LoQ":loq,"blank_sd_A":sd_blank}
                except Exception as e:
                    st.error(f"Linear fit failed: {e}"); return None,None,None
            res.update({
                "n_points": len(xs),
                "levels": sorted(set(xs)),
                "weighting": weighting_scheme,
                "used_replicate_means": use_means
            })
            return (res, png, pdf, xs, ys)


        st.markdown("### Model Fit")
        fit_res, png, pdf = fit(sub, grp, f"{analyte} calibration")
        if fit_res:
            # Residuals (linear only)
            residual_png = None
            if fit_res.get("model") == "linear":
                # Rebuild x/y used in the fit for residual calc
                use_means_now = fit_res.get("used_replicate_means", False)
                if use_means_now:
                    xs = grp["conc_mgL"].astype(float).tolist()
                    ys = grp["A_mean"].astype(float).tolist()
                    sds = grp["A_sd"].astype(float).tolist()
                else:
                    xs = sub["conc_mgL"].astype(float).tolist()
                    ys = sub["A"].astype(float).tolist()
                    sds = None
                # weights for LOOCV if variance-weighted selected
                weights = None
                if weighting_scheme=="Variance-weighted (1/SD^2)" and use_means_now:
                    eps=1e-6; nz=[sd for sd in sds if sd and sd>0]; base=np.median(nz) if nz else 0.01
                    denom=[(sd if (sd and sd>0) else base)**2 + eps for sd in sds]
                    weights=[1.0/d for d in denom]
                m = fit_res["m"]; b = fit_res["b"]
                resids, yhat = residuals_linear(xs, ys, m, b)
                residual_png = make_residual_plot(xs, resids, title=f"{analyte} residuals")
                st.markdown("#### Residual Analysis")
                st.dataframe(pd.DataFrame({"Conc_mgL": xs, "A_obs": ys, "A_fit": yhat, "Residual": resids}))
                st.image(residual_png, caption="Residuals vs Conc", use_column_width=True)

                # LOOCV
                cv = loocv_linear(xs, ys, weights=weights)
                st.markdown("#### Leave-One-Out Cross-Validation (Linear)")
                st.json(cv, expanded=False)

            st.json(fit_res, expanded=False)
            if png:
                st.download_button("Download calibration plot (PNG)", data=png, file_name=f"{analyte}_calibration.png", mime="image/png")
                st.download_button("Download calibration plot (PDF)", data=pdf, file_name=f"{analyte}_calibration.pdf", mime="application/pdf")

            # Model export
            model = {
                "analyte": analyte,
                "created_at": datetime.utcnow().isoformat()+"Z",
                "channel": prof["channel"],
                "absorbance_formula": prof["absorbance_formula"],
                "fit": fit_res,
                "range_hint_mgL": [0.0, 25.0]
            }
            st.download_button("Download model JSON", data=json.dumps(model, indent=2), file_name=f"{analyte}_model.json", mime="application/json")

            # Project ZIP export
            proj = export_project_zip(analyte, df, model, cal_png=png, cal_pdf=pdf, residual_png=residual_png)
            st.download_button("Download Project ZIP", data=proj, file_name=f"{analyte}_project.zip", mime="application/zip")

        out = fit(sub, grp, f"{analyte} calibration")
        if out:
            fit_res, png, pdf, xs_fit, ys_fit = out
            st.json(fit_res, expanded=False)
            if png:
                st.download_button("Download calibration plot (PNG)", data=png, file_name=f"{analyte}_calibration.png", mime="image/png")
                st.download_button("Download calibration plot (PDF)", data=pdf, file_name=f"{analyte}_calibration.pdf", mime="application/pdf")
            # Model export
            model = {
                "analyte": analyte,
                "created_at": datetime.utcnow().isoformat()+"Z",
                "channel": prof["channel"],
                "absorbance_formula": prof["absorbance_formula"],
                "fit": fit_res,
                "range_hint_mgL": [0.0, 25.0]
            }
            st.download_button("Download model JSON", data=json.dumps(model, indent=2), file_name=f"{analyte}_model.json", mime="application/json")

            # Residuals, LOOCV, and session ZIP
            if fit_res["model"]=="linear":
                model_for_resid = {"model":"linear","m":fit_res["m"],"b":fit_res["b"]}
                loocv = loocv_metrics(xs_fit, ys_fit, model_type="linear")
            else:
                model_for_resid = {"model":"quadratic","a":fit_res["a"],"b":fit_res["b"],"c":fit_res["c"]}
                loocv = loocv_metrics(xs_fit, ys_fit, model_type="quadratic")
            png_res, pdf_res = plot_residuals(xs_fit, ys_fit, model_for_resid, title=f"{analyte} residuals")
            st.download_button("Download residuals plot (PNG)", data=png_res, file_name=f"{analyte}_residuals.png", mime="image/png")
            st.download_button("Download residuals plot (PDF)", data=pdf_res, file_name=f"{analyte}_residuals.pdf", mime="application/pdf")
            st.markdown("**LOOCV diagnostics**")
            st.json(loocv, expanded=False)

            plots = {}
            if png: plots[f"{analyte}_calibration.png"] = png
            if png_res: plots[f"{analyte}_residuals.png"] = png_res
            zip_bytes = export_session_zip(analyte, prof, cal_table=df, model=model, plots=plots)
            st.download_button("Export Session ZIP", data=zip_bytes, file_name=f"{analyte}_session.zip", mime="application/zip")

# =========================
# Unknown Prediction
# =========================

with tabs[1]:
    st.subheader("Unknown Prediction")
    model_file = st.file_uploader("Model JSON", type=["json"], key="model_up")
    runs = st.file_uploader("Run JSON(s)", type=["json"], accept_multiple_files=True, key="run_up")
    if model_file and runs:
        try:
            model = json.loads(model_file.getvalue().decode("utf-8"))
        except Exception:
            model = json.loads(model_file.getvalue())
        chan = model.get("channel","SC_Green")
        formula = model.get("absorbance_formula","log10_bg_over_sample")
        for f in runs:
            A, diag, _ = compute_absorbance(f.getvalue(), chan_key=chan, formula=formula)
            if model["fit"]["model"]=="linear":
                m=model["fit"]["m"]; b=model["fit"]["b"]
                C = predict_conc_linear(A, m, b)
            else:
                a=model["fit"]["a"]; bq=model["fit"]["b"]; c=model["fit"]["c"]
                A0 = c - A
                disc = bq*bq - 4*a*A0
                if disc < 0: C = float("nan")
                else:
                    r1 = (-bq + math.sqrt(disc))/(2*a) if a!=0 else float("nan")
                    r2 = (-bq - math.sqrt(disc))/(2*a) if a!=0 else float("nan")
                    lo,hi = model.get("range_hint_mgL",[0,25]); mid=(lo+hi)/2.0
                    candidates = [r for r in [r1,r2] if np.isfinite(r)]
                    C = min(candidates, key=lambda r: abs(r-mid)) if candidates else float("nan")
            st.write({"file": f.name, "A": A, "conc_mgL": C, **diag})

# =========================
# DOE Designer
# =========================

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

with tabs[2]:
    st.subheader("DOE Designer")
    st.write("Build calibration plans and method-development experiments.")
    design = st.selectbox("Design type", ["Calibration (auto-generator)","Calibration (manual levels)","2-level full factorial (screening)","Central Composite (CCD)"], index=0)

    # Templates by analyte (editable starter sets)
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
    templ = DOE_TEMPLATES.get(analyte, DOE_TEMPLATES.get("Iron"))

    reps = st.number_input("Replicates per condition", 1, 10, 2, 1)
    # Per-analyte DOE factor presets (editable by user)
    presets = {
        "Iron": "pH: 3, 6\nBuffer_mM: 10, 50\nReagent_uL: 200, 1500\nTime_min: 1, 10\nTemp_C: 20, 35",
        "Phosphate": "pH: 5, 9\nMolybdate_mM: 5, 25\nAscorbic_mM: 5, 20\nTime_min: 5, 20\nTemp_C: 20, 35",
        "Ammonia": "pH: 10, 11.5\nNessler_uL: 100, 800\nTime_min: 2, 15\nTemp_C: 20, 35",
    }
    st.markdown("**Preset factors (optional):**")
    preset_txt = presets.get(analyte, presets["Iron"])

    if design == "Calibration (auto-generator)":
        st.markdown("Generate calibration levels from range and scheme.")
        colA, colB, colC = st.columns(3)
        with colA:
            cmin = st.number_input("Min conc (mg/L)", 0.0, 1e9, 0.0, 0.01)
        with colB:
            cmax = st.number_input("Max conc (mg/L)", 0.0, 1e9, 25.0, 0.01)
        with colC:
            npts = st.number_input("# nonzero points", 2, 15, 6, 1)
        scheme = st.selectbox("Spacing", ["Linear", "Logarithmic", "ICH template"], index=0)
        include_blank = st.checkbox("Include blank (0 mg/L)", value=True)
        if st.button("Build auto plan"):
            levels = []
            if scheme == "Linear":
                levels = list(np.linspace(cmin, cmax, npts))
            elif scheme == "Logarithmic":
                low = max(cmin, 1e-6)
                levels = list(np.logspace(np.log10(low), np.log10(max(cmax, low*10)), npts))
            else:  # ICH template: ~5 nonzero + blank at 20,40,60,80,100% of range
                lo, hi = min(cmin, cmax), max(cmin, cmax)
                perc = [0.2, 0.4, 0.6, 0.8, 1.0]
                levels = [lo + p*(hi-lo) for p in perc]
            levels = [round(x, 6) for x in levels if x >= 0]
            seq = ([] if not include_blank else [0.0]) + levels
            seq = seq * reps
            random.shuffle(seq)
            rows = [{"order": i+1, "analyte": analyte, "target_mgL": c, "json_path": "", "notes": ""} for i, c in enumerate(seq)]
            plan = pd.DataFrame(rows); st.dataframe(plan)
            st.download_button("Download DOE CSV", data=plan.to_csv(index=False), file_name=f"{analyte}_calibration_auto_plan.csv", mime="text/csv")

    if design == "Calibration (manual levels)":
        levels_str = st.text_input("Levels (mg/L, comma-separated)", "0, 1, 2, 5, 10, 15, 20, 25")
        if st.button("Build plan"):
            try:
                levels = [float(x.strip()) for x in levels_str.split(",") if x.strip()!=""]
            except Exception:
                st.error("Could not parse levels."); levels = [0,1,2,5,10,15,20,25]
            seq = levels * reps
            random.shuffle(seq)
            rows = [{"order": i+1, "analyte": analyte, "target_mgL": c, "json_path": "", "notes": ""} for i,c in enumerate(seq)]
            plan = pd.DataFrame(rows); st.dataframe(plan)
            st.download_button("Download DOE CSV", data=plan.to_csv(index=False), file_name=f"{analyte}_calibration_plan.csv", mime="text/csv")
    elif design == "2-level full factorial (screening)":
        st.markdown("Enter factors with low/high values (e.g., pH, Buffer_mM, Reagent1_uL, ReactionTime_min, Temperature_C, Wavelength_nm).")
        st.caption("Preset for this analyte (editable):")
        preset_area = st.text_area("Preset factors", preset_txt, key="preset_factorial")

        raw = st.text_area("Factors (one per line as name: low, high)", templ["screening"] if templ else "pH: 3, 6\nBuffer_mM: 10, 50\nReagent1_uL: 100, 1000\nReactionTime_min: 1, 10\nTemperature_C: 20, 35")
        if st.button("Build factorial"):
            factors = {}
            for line in raw.splitlines():
                if ":" in line and "," in line:
                    k, rest = line.split(":",1); lo, hi = rest.split(",",1)
                    try:
                        factors[k.strip()] = [float(lo), float(hi)]
                    except Exception: pass
            if not factors:
                st.error("No valid factors parsed.")
            else:
                df = full_factorial(factors)
                rows = []
                for _ in range(reps):
                    for _, r in df.iterrows():
                        row = {"analyte": analyte}
                        row.update({k: r[k] for k in df.columns if k!="run"})
                        rows.append(row)
                random.shuffle(rows)
                plan = pd.DataFrame(rows); plan.insert(0,"order", range(1,len(rows)+1))
                st.dataframe(plan)
                st.download_button("Download DOE CSV", data=plan.to_csv(index=False), file_name=f"{analyte}_factorial_plan.csv", mime="text/csv")
    else:
        st.markdown("Enter factors with low/high values for CCD.")
        st.caption("Preset for this analyte (editable):")
        preset_area_ccd = st.text_area("Preset factors", preset_txt, key="preset_ccd")

        raw = st.text_area("Factors (one per line as name: low, high)", templ["ccd"] if templ else "pH: 3, 6\nBuffer_mM: 10, 50\nReagent1_uL: 100, 1000\nReactionTime_min: 1, 10")
        center_pts = st.number_input("Center points", 0, 20, 4, 1)
        if st.button("Build CCD"):
            factors = {}
            for line in raw.splitlines():
                if ":" in line and "," in line:
                    k, rest = line.split(":",1); lo, hi = rest.split(",",1)
                    try:
                        factors[k.strip()] = (float(lo), float(hi))
                    except Exception: pass
            if not factors:
                st.error("No valid factors parsed.")
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
                plan = pd.DataFrame(rows); plan.insert(0,"order", range(1,len(rows)+1))
                st.dataframe(plan)
                st.download_button("Download DOE CSV", data=plan.to_csv(index=False), file_name=f"{analyte}_CCD_plan.csv", mime="text/csv")

# =========================
# JSON Explorer
# =========================

with tabs[3]:
    st.subheader("JSON Explorer")
    f = st.file_uploader("Upload a run JSON", type=["json"], key="explore")
    chan = st.text_input("Channel key", value=prof["channel"])
    formula = st.selectbox("Formula", ["log10_bg_over_sample","absorbance_single"], index=["log10_bg_over_sample","absorbance_single"].index(prof["absorbance_formula"]) if prof.get("absorbance_formula") in ["log10_bg_over_sample","absorbance_single"] else 0)
    if f:
        try:
            A, diag, obj = compute_absorbance(f.getvalue(), chan_key=chan, formula=formula)
            st.success(f"A = {A:.6f}")
            st.json(diag, expanded=False)
            locs = get_loc_doses_from_sample(obj)
            if locs: st.write("Detected LOC doses (µL):", locs)
        except Exception as e:
            st.error(str(e))

# =========================
# About / Disclaimer + Session Import
# =========================

with tabs[4]:
    st.markdown("""
# R&D MD and Analysis

A generalized, multi‑analyte app for **method development** and **analysis** of colorimetric/spectrophotometric assays.

**Key features**
- User-defined **Analyte**, channel/wavelength, and absorbance formula.
- Robust JSON parser (Background + Sample in same file; nested `scans` supported).
- **LOC-based spike inference** with **per‑analyte profiles** (roles & stock concentrations).
- Replicate-aware calibration with **weighting** (OLS, 1/max(C,1), **variance‑weighted 1/SD²**).
- Models: Linear or Quadratic; **PNG/PDF** plot export; model JSON export.
- DOE generator: **Calibration set**, **2‑level factorial** (screening), and **Central Composite** (CCD) designs.
- **Residuals** plot and **LOOCV** diagnostics for calibration robustness.
- **Session ZIP** export/import for traceability and collaboration.

### Disclaimer on Analyte Stability
For redox‑active or labile species (e.g., Fe²⁺/Fe³⁺), speciation can change quickly after sampling due to oxidation/reduction, hydrolysis, and precipitation. Delayed analysis without proper preservation may not reflect in‑situ speciation. Acid preservation stabilizes **total analyte** but not speciation. For speciation work, **preserve or complex immediately** at the point of sampling and minimize holding time.

---
""")
    st.markdown("### Session Import")
    zip_up = st.file_uploader("Load Session ZIP", type=["zip"], key="sess_zip")
    if zip_up:
        try:
            loaded = import_session_zip(zip_up.getvalue())
            if loaded.get("profile"):
                set_profile(analyte, loaded["profile"])
            if loaded.get("cal_table") is not None:
                st.session_state["cal_table"] = loaded["cal_table"]
            if loaded.get("model") is not None:
                st.session_state["loaded_model"] = loaded["model"]
            st.success("Session loaded into memory. Switch to other tabs to view.")
        except Exception as e:
            st.error(f"Failed to import session: {e}")
