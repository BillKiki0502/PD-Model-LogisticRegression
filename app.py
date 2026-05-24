import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import itertools
import io
import google.generativeai as genai
from toad.transform import Combiner

# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    artifacts = {}
    
    with open("model_artifacts/combiner_rules.pkl", "rb") as f:
        combiner_rules = pickle.load(f)
    combiner = Combiner()
    combiner.load(combiner_rules)
    artifacts["combiner"] = combiner

    with open("model_artifacts/combiner_final_rules.pkl", "rb") as f:
        combiner_final_rules = pickle.load(f)
    combiner_final = Combiner()
    combiner_final.load(combiner_final_rules)
    artifacts["combiner_final"] = combiner_final

    for name in ["woe", "final_model", "selected_features", "selected_vars_vif", "combination_cols"]:
        with open(f"model_artifacts/{name}.pkl", "rb") as f:
            artifacts[name] = pickle.load(f)
    
    return artifacts

artifacts = load_artifacts()
combiner          = artifacts["combiner"]
combiner_final    = artifacts["combiner_final"]
woe               = artifacts["woe"]
final_model       = artifacts["final_model"]
selected_features = artifacts["selected_features"]
selected_vars_vif = artifacts["selected_vars_vif"]
combination_cols  = artifacts["combination_cols"]

# ============================================================
# SESSION STATE INIT
# ============================================================

if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "pred_prob" not in st.session_state:
    st.session_state.pred_prob = None
if "pred_label" not in st.session_state:
    st.session_state.pred_label = None
if "input_features" not in st.session_state:
    st.session_state.input_features = None
if "ai_narrative" not in st.session_state:
    st.session_state.ai_narrative = None

# ============================================================
# PSI BINS
# ============================================================

PSI_BINS   = [0.0288, 0.0648, 0.0814, 0.0957, 0.11, 0.126, 0.145, 0.167, 0.195, 0.243, 0.619]
BIN_LABELS = [f"Bin {i}" for i in range(1, 11)]

def assign_bin(p):
    for i, (lo, hi) in enumerate(zip(PSI_BINS[:-1], PSI_BINS[1:]), start=1):
        if lo <= p <= hi:
            if i <= 3:
                return f"🟢 Bin {i}"
            elif i <= 8:
                return f"🟡 Bin {i}"
            else:
                return f"🔴 Bin {i}"
    return "Out of range"


# ============================================================
# GEN AI — RISK NARRATIVE
# ============================================================

def generate_ai_explanation(features: dict, pd_score: float, risk_bin: str) -> str:
    """Call Gemini API to generate a plain-language risk narrative."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return "_(Gen AI explanation tidak tersedia — GEMINI_API_KEY belum di-set di Streamlit Secrets.)_"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        drivers = []
        if features.get("dti", 0) > 25:
            drivers.append(f"DTI sangat tinggi ({features['dti']:.1f}%) — beban utang besar relatif pendapatan")
        elif features.get("dti", 0) > 15:
            drivers.append(f"DTI moderat ({features['dti']:.1f}%)")
        if features.get("inq_last_6mths", 0) >= 3:
            drivers.append(f"banyak inquiry kredit ({features['inq_last_6mths']}x dalam 6 bulan)")
        if features.get("acc_now_delinq", 0) > 0:
            drivers.append(f"ada {features['acc_now_delinq']} akun yang saat ini menunggak")
        if features.get("emp_length_int", 10) < 2:
            drivers.append(f"riwayat kerja pendek ({features['emp_length_int']} tahun)")
        if features.get("mths_since_earliest_cr_line", 999) < 36:
            drivers.append("riwayat kredit masih sangat baru (< 3 tahun)")
        if features.get("annual_inc", 999999) < 30000:
            drivers.append(f"pendapatan tahunan rendah (${features['annual_inc']:,.0f})")

        drivers_text = "; ".join(drivers) if drivers else "tidak ada faktor risiko dominan yang teridentifikasi"

        prompt = f"""Kamu adalah analis kredit senior. Tulis narasi risiko singkat (3–4 kalimat) untuk peminjam berikut, ditujukan kepada loan officer (bukan teknis).

Data peminjam:
- Probability of Default: {pd_score:.1%}
- Risk Category: {risk_bin}
- Tujuan pinjaman: {features.get('purpose', '-')}
- Kepemilikan rumah: {features.get('home_ownership', '-')}
- Pendapatan tahunan: ${features.get('annual_inc', 0):,.0f}
- DTI: {features.get('dti', 0):.1f}%
- Lama bekerja: {features.get('emp_length_int', 0)} tahun
- Inquiry 6 bulan: {features.get('inq_last_6mths', 0)}
- Faktor risiko utama: {drivers_text}

Akhiri dengan satu rekomendasi singkat: Approve / Review lebih lanjut / Decline (boleh dengan kondisi).
Tulis dalam Bahasa Indonesia yang profesional dan mudah dipahami."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"_(Error saat memanggil Gen AI: {e})_"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_features(df):
    df = df.copy()
    df['sisa_kapasitas'] = (df['annual_inc'] / 12) * (1 - (df['dti'] / 100))

    credit_hist = df['mths_since_earliest_cr_line']
    denom = np.where(
        credit_hist.isna() | (credit_hist == 0),
        credit_hist.fillna(0) + 1,
        credit_hist
    )

    df['employment_credit_ratio'] = df['emp_length_int'] / (denom / 12)
    df['delinq_per_credit_year']  = df['acc_now_delinq'] / (denom / 12)
    df['credit_stress_index']     = df['inq_last_6mths'] * df['dti']
    df['stability_stress']        = df['dti'] / np.maximum(df['emp_length_int'], 1)
    df['cr_age_at_issue']         = df['mths_since_earliest_cr_line'] - df['mths_since_issue_d']
    df['inq_pressure']            = df['inq_last_6mths'] / np.maximum(df['mths_since_earliest_cr_line'] / 12, 1)
    return df


def create_bin_combinations(df, cols, combination_size=2):
    new_features = {}
    for combo in itertools.combinations(sorted(cols), combination_size):
        col_name = "_X_".join(combo)
        new_features[col_name] = (
            df[list(combo)].astype(str).agg("_".join, axis=1)
        )
    new_df = pd.DataFrame(new_features)
    return new_df, list(new_df.columns)


MODEL_FEATURES = [
    'home_ownership', 'purpose', 'verification_status', 'term',
    'emp_length_int', 'mths_since_issue_d', 'mths_since_earliest_cr_line',
    'acc_now_delinq', 'inq_last_6mths', 'annual_inc', 'dti',
    'sisa_kapasitas', 'employment_credit_ratio', 'delinq_per_credit_year',
    'credit_stress_index', 'stability_stress', 'cr_age_at_issue', 'inq_pressure'
]


def predict_pipeline(df_input: pd.DataFrame) -> pd.DataFrame:
    df = create_features(df_input)
    binned      = combiner.transform(df, labels=True)
    combined, _ = create_bin_combinations(binned, combination_cols, combination_size=2)
    full        = pd.concat([df[MODEL_FEATURES], combined], axis=1)
    final       = combiner_final.transform(full, labels=True)
    selected    = final[selected_vars_vif].copy()
    woe_transformed = woe.transform(selected)
    X           = sm.add_constant(woe_transformed[selected_features], has_constant='add')
    prob        = final_model.predict(X)

    result = df_input.copy()
    result["predicted_prob"]  = prob.values
    result["predicted_label"] = (prob >= 0.13).astype(int)
    result["risk_category"]   = result["predicted_prob"].apply(assign_bin)
    return result


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Default Probability Predictor")
st.markdown("Model berbasis **Logistic Regression + WOE Binning** untuk prediksi risiko kredit.")

tab1, tab2 = st.tabs(["📝 Input Manual", "📂 Upload CSV"])

# ──────────────────────────────────────────────────────────────
# TAB 1 — INPUT MANUAL
# ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Isi data peminjam secara manual")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Informasi Dasar**")
        home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
            help="Status kepemilikan rumah"
        )
        purpose = st.selectbox(
            "Purpose",
            ["debt_consolidation", "credit_card", "home_improvement",
             "other", "major_purchase", "small_business", "car",
             "medical", "moving", "vacation", "house", "wedding",
             "renewable_energy", "educational"],
            help="Tujuan pinjaman"
        )
        verification_status = st.selectbox(
            "Verification Status",
            ["Not Verified", "Verified", "Source Verified"],
            help="Status verifikasi pendapatan"
        )
        term = st.selectbox(
            "Term (bulan)",
            [36, 60],
            help="Tenor pinjaman"
        )

    with col2:
        st.markdown("**💼 Pekerjaan & Kredit**")
        emp_length_int = st.slider(
            "Employment Length (tahun)", 0, 10, 3,
            help="Lama bekerja"
        )
        mths_since_issue_d = st.number_input(
            "Months Since Issue Date", min_value=0, max_value=300, value=36,
            help="Berapa bulan sejak pinjaman diterbitkan"
        )
        mths_since_earliest_cr_line = st.number_input(
            "Months Since Earliest Credit Line", min_value=0, max_value=600, value=120,
            help="Usia kredit tertua (bulan)"
        )
        inq_last_6mths = st.number_input(
            "Inquiries Last 6 Months", min_value=0, max_value=20, value=1,
            help="Jumlah inquiry kredit 6 bulan terakhir"
        )

    with col3:
        st.markdown("**💰 Keuangan**")
        annual_inc = st.number_input(
            "Annual Income (USD)", min_value=0.0, value=60000.0, step=1000.0,
            help="Pendapatan tahunan"
        )
        dti = st.number_input(
            "Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1,
            help="Rasio utang terhadap pendapatan"
        )
        acc_now_delinq = st.number_input(
            "Accounts Now Delinquent", min_value=0, max_value=20, value=0,
            help="Jumlah akun yang sedang menunggak"
        )

    st.divider()

    # ── Tombol Prediksi ───────────────────────────────────────
    if st.button("🔍 Prediksi Risiko", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "home_ownership":              home_ownership,
            "purpose":                     purpose,
            "verification_status":         verification_status,
            "term":                        term,
            "emp_length_int":              emp_length_int,
            "mths_since_issue_d":          mths_since_issue_d,
            "mths_since_earliest_cr_line": mths_since_earliest_cr_line,
            "acc_now_delinq":              acc_now_delinq,
            "inq_last_6mths":              inq_last_6mths,
            "annual_inc":                  annual_inc,
            "dti":                         dti,
        }])

        with st.spinner("Memproses prediksi..."):
            try:
                result = predict_pipeline(input_data)
                # Simpan ke session_state supaya tidak hilang saat button lain diklik
                st.session_state.pred_prob    = result["predicted_prob"].iloc[0]
                st.session_state.pred_label   = result["risk_category"].iloc[0]
                st.session_state.input_features = {
                    "home_ownership":              home_ownership,
                    "purpose":                     purpose,
                    "annual_inc":                  annual_inc,
                    "dti":                         dti,
                    "emp_length_int":              emp_length_int,
                    "inq_last_6mths":              inq_last_6mths,
                    "acc_now_delinq":              acc_now_delinq,
                    "mths_since_earliest_cr_line": mths_since_earliest_cr_line,
                }
                st.session_state.show_result  = True
                st.session_state.ai_narrative = None  # reset narasi lama
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
                st.exception(e)

    # ── Tampilkan Hasil — DI LUAR if button, pakai session_state ──
    if st.session_state.show_result:
        prob  = st.session_state.pred_prob
        label = st.session_state.pred_label

        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        m1.metric("Probability of Default", f"{prob:.2%}")
        m2.metric("Risk Category", label)
        m3.metric("Threshold Used", "0.13")

        st.markdown("**Probability Score:**")
        st.progress(min(float(prob), 1.0))

        bin_num = int(label.split("Bin ")[-1]) if "Bin" in label else 0
        if bin_num >= 9:
            st.error("⚠️ Peminjam ini memiliki risiko tinggi gagal bayar.")
        elif bin_num >= 4:
            st.warning("⚠️ Peminjam ini berada di zona risiko sedang.")
        else:
            st.success("✅ Peminjam ini memiliki risiko rendah.")

        # ── Gen AI Explanation ───────────────────────────────
        st.markdown("---")
        st.subheader("🤖 AI Risk Narrative")
        st.caption("Penjelasan otomatis berbasis Gen AI untuk membantu loan officer.")

        if st.button("✨ Generate AI Explanation", use_container_width=True):
            with st.spinner("Membuat narasi risiko..."):
                st.session_state.ai_narrative = generate_ai_explanation(
                    st.session_state.input_features, prob, label
                )

        if st.session_state.ai_narrative:
            st.info(st.session_state.ai_narrative)

# ──────────────────────────────────────────────────────────────
# TAB 2 — UPLOAD CSV
# ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload file CSV untuk prediksi batch")

    st.info(
        "**Kolom yang diperlukan:** `home_ownership`, `purpose`, `verification_status`, "
        "`term`, `emp_length_int`, `mths_since_issue_d`, `mths_since_earliest_cr_line`, "
        "`acc_now_delinq`, `inq_last_6mths`, `annual_inc`, `dti`"
    )

    template_cols = [
        "home_ownership", "purpose", "verification_status", "term",
        "emp_length_int", "mths_since_issue_d", "mths_since_earliest_cr_line",
        "acc_now_delinq", "inq_last_6mths", "annual_inc", "dti"
    ]
    sample_data = pd.DataFrame([{
        "home_ownership": "RENT", "purpose": "debt_consolidation",
        "verification_status": "Verified", "term": 36,
        "emp_length_int": 3, "mths_since_issue_d": 36,
        "mths_since_earliest_cr_line": 120, "acc_now_delinq": 0,
        "inq_last_6mths": 1, "annual_inc": 60000.0, "dti": 15.0
    }])

    st.download_button(
        "⬇️ Download Template CSV",
        data=sample_data.to_csv(index=False),
        file_name="template_input.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV kamu di sini", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"**Preview data ({len(df_upload)} baris):**")
            st.dataframe(df_upload.head(5), use_container_width=True)

            if st.button("🚀 Prediksi Semua", type="primary", use_container_width=True):
                with st.spinner(f"Memproses {len(df_upload)} baris..."):
                    result_df = predict_pipeline(df_upload)

                    total     = len(result_df)
                    high_risk = (result_df["predicted_label"] == 1).sum()
                    low_risk  = total - high_risk

                    st.success(f"✅ Prediksi selesai untuk {total} data")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Data", total)
                    c2.metric("🔴 Predicted Default", high_risk, f"{high_risk/total:.1%}")
                    c3.metric("🟢 Predicted Non-Default", low_risk, f"{low_risk/total:.1%}")

                    st.markdown("**Hasil Prediksi:**")
                    st.dataframe(
                        result_df[["predicted_prob", "predicted_label", "risk_category"]
                                  + template_cols].head(100),
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Download Hasil Prediksi (CSV)",
                        data=result_df.to_csv(index=False),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error memproses file: {e}")
            st.exception(e)

# ──────────────────────────────────────────────────────────────
# SIDEBAR — info model
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Info Model")
    st.markdown("""
    **Model:** Logistic Regression (statsmodels)
    
    **Pipeline:**
    1. Feature Engineering
    2. Auto Binning (Combiner #1)
    3. Bin Combination
    4. Auto Binning (Combiner #2)
    5. IV / Correlation / VIF Filter
    6. WOE Transformation
    7. Stepwise Selection
    8. Logistic Regression
    
    **Threshold:** 0.13
    
    **Risk Bins (dari PSI table):**
    - 🟢 Bin 1–3 → Low Risk  (prob ≤ 0.10)
    - 🟡 Bin 4–7 → Medium Risk (0.10 – 0.20)
    - 🔴 Bin 9–10 → High Risk (prob > 0.20)
    
    ---
    **🤖 Gen AI Explanation:**
    Tersedia di tab Input Manual setelah prediksi.
    Membutuhkan `GEMINI_API_KEY` di Streamlit Secrets.
    """)
