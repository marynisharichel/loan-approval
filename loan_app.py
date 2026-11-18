import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSense AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — Dark Luxury Finance Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0d14;
    color: #e8e4dc;
}
.stApp { background: linear-gradient(135deg, #0a0d14 0%, #111827 50%, #0d1117 100%); }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c9a84c !important;
    font-family: 'DM Serif Display', serif;
}

h1 { font-family: 'DM Serif Display', serif !important; font-size: 2.6rem !important; color: #e8e4dc !important; letter-spacing: -0.5px; }
h2 { font-family: 'DM Serif Display', serif !important; color: #c9a84c !important; font-size: 1.5rem !important; }
h3 { font-family: 'DM Sans', sans-serif !important; color: #94a3b8 !important; font-weight: 500 !important; font-size: 1.0rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

.stSlider > div > div > div { background: #c9a84c !important; }
.stNumberInput input, .stSelectbox select, [data-baseweb="select"] {
    background: #1a2235 !important;
    border: 1px solid #2a3a55 !important;
    color: #e8e4dc !important;
    border-radius: 8px !important;
}
.stTextInput input { background: #1a2235 !important; border: 1px solid #2a3a55 !important; color: #e8e4dc !important; }
[data-baseweb="select"] > div { background: #1a2235 !important; border-color: #2a3a55 !important; color: #e8e4dc !important; }

.stButton > button {
    background: linear-gradient(135deg, #c9a84c 0%, #a07830 100%);
    color: #0a0d14;
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.85rem;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(201, 168, 76, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(201, 168, 76, 0.45);
    background: linear-gradient(135deg, #d4b55a 0%, #b08840 100%);
}

.card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.card-gold { border-color: #c9a84c44; }
.card-green { border-color: #22c55e44; }
.card-red { border-color: #ef444444; }
.card-amber { border-color: #f59e0b44; }
.card-blue { border-color: #3b82f644; }

.badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-green { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e55; }
.badge-red { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
.badge-amber { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }

.gauge-wrap { position: relative; text-align: center; }
.prob-big { font-size: 3.5rem; font-family: 'DM Serif Display', serif; line-height: 1; }
.prob-label { font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin-top: 4px; }

.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-box {
    flex: 1; min-width: 120px;
    background: #1a2235;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 700; font-family: 'DM Serif Display', serif; color: #c9a84c; }
.metric-lbl { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

.factor-bar-wrap { margin-bottom: 0.7rem; }
.factor-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 3px; }
.factor-bar-bg { background: #1a2235; border-radius: 4px; height: 8px; overflow: hidden; }
.factor-bar-fill { height: 8px; border-radius: 4px; transition: width 0.8s ease; }

.gold-divider { border: none; border-top: 1px solid #c9a84c33; margin: 1.5rem 0; }

.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748b; border-radius: 8px;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] { background: #1e2d45 !important; color: #c9a84c !important; }

.stMarkdown p { color: #94a3b8; }
label, .stSlider label, .stNumberInput label, .stSelectbox label { color: #94a3b8 !important; font-size: 0.85rem !important; }
.element-container { margin-bottom: 0.3rem; }
div[data-testid="stHorizontalBlock"] { gap: 16px; }

/* Info pill */
.info-pill {
    display: inline-block;
    background: #1e2d45;
    border: 1px solid #3b82f655;
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    color: #93c5fd;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATASET & MODEL TRAINING
# ─────────────────────────────────────────────
DATASET_PATH =r"C:\Users\maryn\Desktop\minor project\loan_approval_dataset.csv"

@st.cache_resource
def train_model():
    df_raw = pd.read_csv(DATASET_PATH)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw["loan_status"] = df_raw["loan_status"].str.strip()

    df = pd.DataFrame()
    df["income"]        = df_raw["income_annum"]
    df["loan_amount"]   = df_raw["loan_amount"]
    df["loan_term"]     = df_raw["loan_term"]
    df["cibil"]         = df_raw["cibil_score"]
    df["dependents"]    = df_raw["no_of_dependents"]
    df["education"]     = (df_raw["education"].str.strip() == "Graduate").astype(int)
    df["self_employed"] = (df_raw["self_employed"].str.strip() == "Yes").astype(int)
    df["res_assets"]    = df_raw["residential_assets_value"]
    df["com_assets"]    = df_raw["commercial_assets_value"]
    df["lux_assets"]    = df_raw["luxury_assets_value"]
    df["bank_assets"]   = df_raw["bank_asset_value"]
    df["total_assets"]  = df["res_assets"] + df["com_assets"] + df["lux_assets"] + df["bank_assets"]

    # ── Binned ratios to avoid perfect leakage ──
    df["income_ratio"] = pd.cut(
        df["loan_amount"] / df["income"].clip(lower=1),
        bins=[0, 1, 2, 3, 5, 100], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["asset_ratio"] = pd.cut(
        df["total_assets"] / df["loan_amount"].clip(lower=1),
        bins=[0, 0.5, 1, 2, 5, 1000], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["approved"] = (df_raw["loan_status"] == "Approved").astype(int)

    features = ["income", "loan_amount", "loan_term", "cibil", "dependents",
                "education", "self_employed", "total_assets", "income_ratio", "asset_ratio"]

    X = df[features]
    y = df["approved"]

    # Stratified split for balanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Regularised models ──
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        min_samples_leaf=10, max_features="sqrt",
        random_state=42
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05,
        max_depth=3, min_samples_leaf=10,
        subsample=0.8, random_state=42
    )
    lr = LogisticRegression(max_iter=500, C=0.8, random_state=42)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    lr.fit(X_train_s, y_train)

    # ── Honest accuracy: 5-fold cross-validation ──
    cv_rf = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()
    cv_gb = cross_val_score(gb, X, y, cv=5, scoring="accuracy").mean()
    cv_lr = cross_val_score(lr, scaler.fit_transform(X), y, cv=5, scoring="accuracy").mean()

    

    importances = rf.feature_importances_

   

    return rf, gb, lr, scaler, features, importances, cv_rf, cv_gb, cv_lr, df

rf_model, gb_model, lr_model, scaler, feature_names, importances, \
    cv_rf, cv_gb, cv_lr, train_df = train_model()


def predict_loan(income, loan_amount, loan_term, cibil, dependents,
                 education_val, self_employed_val, res, com, lux, bank):
    total_assets = res + com + lux + bank

    income_ratio_raw = loan_amount / max(income, 1)
    asset_ratio_raw  = total_assets / max(loan_amount, 1)

    ir_bin = pd.cut([income_ratio_raw], bins=[0,1,2,3,5,100], labels=[0,1,2,3,4]).astype(int)[0]
    ar_bin = pd.cut([asset_ratio_raw],  bins=[0,0.5,1,2,5,1000], labels=[0,1,2,3,4]).astype(int)[0]

    x = np.array([[income, loan_amount, loan_term, cibil, dependents,
                   education_val, self_employed_val, total_assets, ir_bin, ar_bin]])
    x_s = scaler.transform(x)

    p_rf = rf_model.predict_proba(x)[0][1]
    p_gb = gb_model.predict_proba(x)[0][1]
    p_lr = lr_model.predict_proba(x_s)[0][1]
    prob = p_rf * 0.45 + p_gb * 0.40 + p_lr * 0.15

    if prob >= 0.65:
        decision = "APPROVED"
    elif prob >= 0.40:
        decision = "CONDITIONAL"
    else:
        decision = "REJECTED"

    return prob, decision, p_rf, p_gb, p_lr, total_assets, income_ratio_raw, asset_ratio_raw


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💎 LoanSense AI")
    st.markdown("*Precision lending intelligence*")
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    st.markdown("### 👤 Personal Profile")
    income      = st.number_input("Annual Income (₹)", min_value=0, value=600000, step=50000)
    dependents  = st.number_input("Dependents", min_value=0, max_value=20, value=1, step=1)
    education   = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Employment", ["Salaried", "Self-Employed"])

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    st.markdown("### 🏦 Loan Details")
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=1500000, step=100000)
    loan_term   = st.number_input("Loan Term (Months)", min_value=6, max_value=480, value=120, step=12)
    cibil       = st.slider("CIBIL Score", 300, 900, 720)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    st.markdown("### 🏠 Assets (₹)")
    res  = st.number_input("Residential",   min_value=0, value=2000000, step=100000)
    com  = st.number_input("Commercial",    min_value=0, value=500000,  step=100000)
    lux  = st.number_input("Luxury",        min_value=0, value=200000,  step=50000)
    bank = st.number_input("Bank/Financial", min_value=0, value=300000, step=50000)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    predict_btn = st.button("⚡ Analyse Application", use_container_width=True)

    # ── Model accuracy panel with honest CV scores ──
    st.markdown("### 📈 Model Performance")
    st.markdown(f"""
    <div style="margin-bottom:0.6rem;">
      <span class="info-pill">Scores validated with 5-fold cross-validation</span>
    </div>

    <div style="font-size:0.72rem; color:#c9a84c; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">Cross-Validation Accuracy</div>
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-val">{cv_rf*100:.1f}%</div>
        <div class="metric-lbl" style="font-size:0.65rem;">Random Forest</div>
      </div>
      <div class="metric-box">
        <div class="metric-val">{cv_gb*100:.1f}%</div>
        <div class="metric-lbl" style="font-size:0.65rem;">Gradient Boost</div>
      </div>
    </div>
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-val">{cv_lr*100:.1f}%</div>
        <div class="metric-lbl" style="font-size:0.65rem;">Logistic Reg.</div>
      </div>
      

    
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
edu_val = 1 if education == "Graduate" else 0
emp_val = 1 if self_employed == "Self-Employed" else 0

st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:0.5rem;">
  <div>
    <h1 style="margin:0">LoanSense AI</h1>
    <p style="color:#64748b; margin:4px 0 0; font-size:1.0rem;">ML-powered loan eligibility engine • Trained on 4,269 real-world profiles</p>
  </div>
</div>

""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Simulators", "📉 Feature Insights", "🗄️ Dataset Stats"])

# ═══════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════
with tab1:
    if predict_btn or "last_result" in st.session_state:
        if predict_btn:
            prob, decision, p_rf, p_gb, p_lr, total_assets, income_ratio, asset_ratio = predict_loan(
                income, loan_amount, loan_term, cibil, dependents,
                edu_val, emp_val, res, com, lux, bank
            )
            st.session_state["last_result"] = (prob, decision, p_rf, p_gb, p_lr, total_assets, income_ratio, asset_ratio)
        else:
            prob, decision, p_rf, p_gb, p_lr, total_assets, income_ratio, asset_ratio = st.session_state["last_result"]

        pct = int(prob * 100)
        badge_cls  = "badge-green" if decision=="APPROVED" else ("badge-amber" if decision=="CONDITIONAL" else "badge-red")
        badge_text = {"APPROVED":"✅ Approved","CONDITIONAL":"⚠️ Conditional","REJECTED":"❌ Rejected"}[decision]
        prob_color = "#22c55e" if decision=="APPROVED" else ("#f59e0b" if decision=="CONDITIONAL" else "#ef4444")
        card_cls   = "green" if decision=="APPROVED" else ("amber" if decision=="CONDITIONAL" else "red")

        col_a, col_b = st.columns([1, 1.6], gap="large")

        with col_a:
            st.markdown(f"""
            <div class="card card-{card_cls}">
              <div class="gauge-wrap">
                <div class="prob-big" style="color:{prob_color}">{pct}<span style="font-size:1.8rem">%</span></div>
                <div class="prob-label">Approval Probability</div>
                <div style="margin: 1rem 0 0.5rem;">
                  <span class="badge {badge_cls}">{badge_text}</span>
                </div>
              </div>
              <hr class="gold-divider">
              <div style="font-size:0.72rem; color:#475569; text-align:center; margin-bottom:0.7rem;">Individual model votes</div>
              <div class="metric-row">
                <div class="metric-box"><div class="metric-val" style="font-size:1.1rem">{p_rf*100:.0f}%</div><div class="metric-lbl">RF</div></div>
                <div class="metric-box"><div class="metric-val" style="font-size:1.1rem">{p_gb*100:.0f}%</div><div class="metric-lbl">GBM</div></div>
                <div class="metric-box"><div class="metric-val" style="font-size:1.1rem">{p_lr*100:.0f}%</div><div class="metric-lbl">LR</div></div>
              </div>
              <div style="font-size:0.7rem; color:#475569; text-align:center">RF 45% · GBM 40% · LR 15% ensemble</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("## Risk Factor Breakdown")
            factors = {
                "Credit Score (CIBIL)":  min(cibil / 900, 1.0),
                "Income Adequacy":        min(1 / max(income_ratio, 0.1), 1.0),
                "Asset Coverage":         min(asset_ratio / 2, 1.0),
                "Loan-to-Term Fit":       1 - min(loan_term / 480, 1.0),
                "Profile Stability":      (edu_val * 0.5 + (1 - emp_val) * 0.5),
                "Dependency Load":        max(0, 1 - dependents / 6),
            }
            for name, score_val in factors.items():
                bar_pct   = int(score_val * 100)
                bar_color = "#22c55e" if score_val > 0.65 else ("#f59e0b" if score_val > 0.4 else "#ef4444")
                st.markdown(f"""
                <div class="factor-bar-wrap">
                  <div style="display:flex; justify-content:space-between;">
                    <span class="factor-label">{name}</span>
                    <span style="font-size:0.75rem; color:#64748b">{bar_pct}%</span>
                  </div>
                  <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:{bar_pct}%; background:{bar_color}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

        rec_col1, rec_col2 = st.columns(2)
        reasons, suggestions = [], []
        if cibil < 650:       reasons.append("Low CIBIL score (below 650)")
        if income_ratio > 3:  reasons.append(f"Loan is {income_ratio:.1f}× your annual income")
        if asset_ratio < 0.5: reasons.append("Assets below 50% of loan value")
        if dependents > 3:    reasons.append(f"{dependents} dependents increase monthly burden")
        if emp_val:           reasons.append("Self-employed income flagged as variable")
        if loan_term > 240:   reasons.append("Repayment span exceeds 20 years")

        if cibil < 750:       suggestions.append("Raise CIBIL above 750 for best rates")
        if income_ratio > 2.5:suggestions.append("Reduce loan or show supplemental income")
        if asset_ratio < 1.0: suggestions.append("Offer additional collateral")
        if emp_val:           suggestions.append("Provide 3-year ITR to strengthen profile")

        with rec_col1:
            st.markdown(f"""
            <div class="card card-red">
              <h3 style="color:#ef4444 !important">⚠️ Risk Flags</h3>
              {''.join(f'<p style="color:#fca5a5; margin:4px 0; font-size:0.88rem">• {r}</p>' for r in reasons) if reasons else '<p style="color:#22c55e; font-size:0.88rem">• No major risk flags detected</p>'}
            </div>
            """, unsafe_allow_html=True)

        with rec_col2:
            st.markdown(f"""
            <div class="card card-gold">
              <h3 style="color:#c9a84c !important">💡 Action Items</h3>
              {''.join(f'<p style="color:#fde68a; margin:4px 0; font-size:0.88rem">→ {s}</p>' for s in suggestions) if suggestions else '<p style="color:#22c55e; font-size:0.88rem">→ Maintain your excellent profile</p>'}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem;">
          <div style="font-size:4rem; margin-bottom:1rem">💎</div>
          <h2>Ready to Analyse</h2>
          <p style="color:#64748b">Fill in your details in the sidebar and click <strong style="color:#c9a84c">Analyse Application</strong> to get your ML-powered prediction.</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2 — SIMULATORS
# ═══════════════════════════════════════════
with tab2:
    st.markdown("## 🧮 Financial Simulators")

    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
        border: 1px solid #c9a84c44;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    sim1, sim2, sim3 = st.columns(3, gap="large")

    with sim1:
        st.markdown("### 📅 EMI Calculator")
        emi_loan   = st.number_input("Loan (₹)", value=int(loan_amount), step=100000, key="emi_loan")
        emi_rate   = st.slider("Annual Interest Rate (%)", 6.0, 24.0, 9.5, 0.1, key="emi_rate")
        emi_months = st.number_input("Tenure (months)", value=int(loan_term), step=12, min_value=12, key="emi_months")

        if emi_loan > 0 and emi_months > 0:
            r = (emi_rate / 100) / 12
            emi = emi_loan * r * (1+r)**emi_months / ((1+r)**emi_months - 1) if r > 0 else emi_loan / emi_months
            total_pay = emi * emi_months
            total_int = total_pay - emi_loan
            int_pct   = (total_int / emi_loan) * 100
            bar_color = "#ef4444" if int_pct > 60 else "#f59e0b" if int_pct > 35 else "#22c55e"

            st.markdown(f"""
            <div style="margin-top:1rem;">
              <div class="metric-box" style="margin-bottom:10px; text-align:left; padding:1rem 1.2rem;">
                <div class="metric-val">₹{emi:,.0f}</div>
                <div class="metric-lbl">Monthly EMI</div>
              </div>
              <div class="metric-row">
                <div class="metric-box"><div class="metric-val" style="font-size:1rem; color:#94a3b8">₹{total_int/100000:.1f}L</div><div class="metric-lbl">Total Interest</div></div>
                <div class="metric-box"><div class="metric-val" style="font-size:1rem; color:#94a3b8">₹{total_pay/100000:.1f}L</div><div class="metric-lbl">Total Payable</div></div>
              </div>
              <div style="margin-top:8px;">
                <div class="factor-label">Interest burden</div>
                <div class="factor-bar-bg">
                  <div class="factor-bar-fill" style="width:{min(int_pct,100):.0f}%; background:{bar_color}"></div>
                </div>
                <div style="font-size:0.72rem; color:#475569; margin-top:2px">{int_pct:.1f}% of principal</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with sim2:
        st.markdown("### 🔬 What-If Simulator")
        st.markdown('<p style="font-size:0.8rem; color:#64748b">Adjust variables and see impact on approval probability</p>', unsafe_allow_html=True)

        wi_cibil  = st.slider("What if CIBIL →", 300, 900, cibil, key="wi_cibil")
        wi_loan   = st.number_input("What if Loan (₹) →", value=int(loan_amount), step=100000, key="wi_loan")
        wi_income = st.number_input("What if Income (₹) →", value=int(income), step=50000, key="wi_income")

        p_orig, _, _, _, _, _, _, _ = predict_loan(income, loan_amount, loan_term, cibil, dependents, edu_val, emp_val, res, com, lux, bank)
        p_new, d_new, _, _, _, _, _, _ = predict_loan(wi_income, wi_loan, loan_term, wi_cibil, dependents, edu_val, emp_val, res, com, lux, bank)

        delta = (p_new - p_orig) * 100
        delta_color = "#22c55e" if delta >= 0 else "#ef4444"
        d_badge = {"APPROVED":"badge-green","CONDITIONAL":"badge-amber","REJECTED":"badge-red"}[d_new]
        d_text  = {"APPROVED":"✅ Approved","CONDITIONAL":"⚠️ Conditional","REJECTED":"❌ Rejected"}[d_new]

        st.markdown(f"""
        <div style="margin-top:1rem; text-align:center;">
          <div class="metric-val" style="font-size:2.5rem; color:{delta_color}">{'+' if delta>=0 else ''}{delta:.1f}%</div>
          <div class="metric-lbl">Change in approval probability</div>
          <div style="margin:0.8rem 0"><span class="badge {d_badge}">{d_text}</span></div>
          <div style="display:flex; justify-content:space-around; margin-top:1rem;">
            <div><div style="color:#64748b; font-size:0.75rem">BEFORE</div><div style="color:#94a3b8; font-size:1.3rem; font-weight:700">{p_orig*100:.0f}%</div></div>
            <div style="font-size:1.5rem; color:#475569">→</div>
            <div><div style="color:#64748b; font-size:0.75rem">AFTER</div><div style="font-size:1.3rem; font-weight:700; color:{delta_color}">{p_new*100:.0f}%</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with sim3:
        st.markdown("### 📈 CIBIL Impact Planner")
        st.markdown('<p style="font-size:0.8rem; color:#64748b">See how improving your credit score changes outcomes</p>', unsafe_allow_html=True)

        cibil_targets = [500, 550, 600, 650, 700, 750, 800, 850]
        rows = []
        for target in cibil_targets:
            p, d, _, _, _, _, _, _ = predict_loan(income, loan_amount, loan_term, target, dependents, edu_val, emp_val, res, com, lux, bank)
            rows.append((target, int(p*100), d))

        bars_html = ""
        for score_t, prob_t, dec_t in rows:
            bar_c  = "#22c55e" if dec_t=="APPROVED" else ("#f59e0b" if dec_t=="CONDITIONAL" else "#ef4444")
            marker = " ◄ You" if score_t <= cibil < score_t + 50 else ""
            bars_html += f"""
            <div class="factor-bar-wrap" style="margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between;">
                <span class="factor-label">{score_t}{marker}</span>
                <span style="font-size:0.75rem; color:{bar_c}">{prob_t}%</span>
              </div>
              <div class="factor-bar-bg">
                <div class="factor-bar-fill" style="width:{prob_t}%; background:{bar_c}"></div>
              </div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 3 — FEATURE INSIGHTS
# ═══════════════════════════════════════════
with tab3:
    st.markdown("## 📉 Model Feature Importance")
    st.markdown('<p style="color:#64748b">Contribution of each feature to the Random Forest model\'s predictions</p>', unsafe_allow_html=True)

    feat_labels = {
        "income": "Annual Income", "loan_amount": "Loan Amount",
        "loan_term": "Loan Term", "cibil": "CIBIL Score",
        "dependents": "Dependents", "education": "Education",
        "self_employed": "Self-Employed", "total_assets": "Total Assets",
        "income_ratio": "Income Ratio (binned)", "asset_ratio": "Asset Ratio (binned)"
    }
    sorted_idx   = np.argsort(importances)[::-1]
    sorted_feats = [(feat_labels[feature_names[i]], importances[i]) for i in sorted_idx]

    col_feat, col_about = st.columns([1.4, 1])

    with col_feat:
        for label, imp in sorted_feats:
            bar_pct   = int(imp * 100 * 7)
            bar_pct   = min(bar_pct, 100)
            imp_color = "#c9a84c" if imp > 0.15 else "#4b9aca" if imp > 0.08 else "#64748b"
            st.markdown(f"""
            <div class="factor-bar-wrap">
              <div style="display:flex; justify-content:space-between;">
                <span class="factor-label">{label}</span>
                <span style="font-size:0.75rem; color:{imp_color}">{imp*100:.1f}%</span>
              </div>
              <div class="factor-bar-bg">
                <div class="factor-bar-fill" style="width:{bar_pct}%; background:{imp_color}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_about:
        top_feat = sorted_feats[0][0]
        st.markdown(f"""
        <div class="card card-gold">
          <h3 style="color:#c9a84c !important">🧠 Model Info</h3>
          <p style="font-size:0.85rem; color:#94a3b8; margin:4px 0">Top predictor: <strong style="color:#e8e4dc">{top_feat}</strong></p>
          <hr class="gold-divider">
          <div class="metric-row">
            <div class="metric-box"><div class="metric-val" style="font-size:1.1rem">{cv_rf*100:.1f}%</div><div class="metric-lbl" style="font-size:0.6rem;">Random Forest CV</div></div>
            <div class="metric-box"><div class="metric-val" style="font-size:1.1rem">{cv_gb*100:.1f}%</div><div class="metric-lbl" style="font-size:0.6rem;">Grad Boost CV</div></div>
          </div>
          <div class="metric-row">
            <div class="metric-box"><div class="metric-val" style="font-size:1.1rem">{cv_lr*100:.1f}%</div><div class="metric-lbl" style="font-size:0.6rem;">Logistic Reg CV</div></div>
            <div class="metric-box"><div class="metric-val" style="font-size:1.1rem">3</div><div class="metric-lbl" style="font-size:0.6rem;">Models</div></div>
          </div>
          <hr class="gold-divider">
          <p style="font-size:0.78rem; color:#475569">Ensemble: RF 45% + GBM 40% + LR 15%<br>Ratios binned to prevent leakage<br>Stratified 80/20 train-test split</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 4 — DATASET STATS
# ═══════════════════════════════════════════
with tab4:
    st.markdown("## 🗄️ Training Dataset Overview")

    approve_rate = train_df["approved"].mean() * 100
    avg_income   = train_df["income"].mean()
    avg_loan     = train_df["loan_amount"].mean()
    avg_cibil    = train_df["cibil"].mean()
    n_profiles   = len(train_df)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box"><div class="metric-val">{n_profiles:,}</div><div class="metric-lbl">Profiles</div></div>
      <div class="metric-box"><div class="metric-val">{approve_rate:.0f}%</div><div class="metric-lbl">Approval Rate</div></div>
      <div class="metric-box"><div class="metric-val">₹{avg_income/100000:.1f}L</div><div class="metric-lbl">Avg Income</div></div>
      <div class="metric-box"><div class="metric-val">₹{avg_loan/100000:.1f}L</div><div class="metric-lbl">Avg Loan</div></div>
      <div class="metric-box"><div class="metric-val">{avg_cibil:.0f}</div><div class="metric-lbl">Avg CIBIL</div></div>
    </div>
    """, unsafe_allow_html=True)

   

    st.markdown("### Sample Records from Training Dataset")
    sample = train_df.sample(10, random_state=99)[
        ["income","loan_amount","cibil","dependents","education","self_employed","total_assets","approved"]
    ].copy()
    sample.columns = ["Income (₹)","Loan (₹)","CIBIL","Dependents","Graduate","Self-Emp","Assets (₹)","Approved"]
    sample["Income (₹)"] = sample["Income (₹)"].map(lambda x: f"₹{x:,.0f}")
    sample["Loan (₹)"]   = sample["Loan (₹)"].map(lambda x: f"₹{x:,.0f}")
    sample["Assets (₹)"] = sample["Assets (₹)"].map(lambda x: f"₹{x:,.0f}")
    sample["Approved"]   = sample["Approved"].map({1:"✅ Yes", 0:"❌ No"})
    sample["Graduate"]   = sample["Graduate"].map({1:"Yes", 0:"No"})
    sample["Self-Emp"]   = sample["Self-Emp"].map({1:"Yes", 0:"No"})
    sample["CIBIL"]      = sample["CIBIL"].map(lambda x: f"{x:.0f}")
    st.dataframe(sample, use_container_width=True, hide_index=True)

    st.markdown("### CIBIL Score Distribution")
    bins = range(300, 950, 50)
    hist_data = []
    for b in list(bins)[:-1]:
        cnt_app = ((train_df["cibil"] >= b) & (train_df["cibil"] < b+50) & (train_df["approved"]==1)).sum()
        cnt_rej = ((train_df["cibil"] >= b) & (train_df["cibil"] < b+50) & (train_df["approved"]==0)).sum()
        hist_data.append((f"{b}–{b+50}", cnt_app, cnt_rej))

    max_cnt = max(h[1]+h[2] for h in hist_data)
    bars_html = ""
    for label, cnt_app, cnt_rej in hist_data:
        w_app = int(cnt_app / max_cnt * 100)
        w_rej = int(cnt_rej / max_cnt * 100)
        bars_html += f"""
        <div class="factor-bar-wrap" style="margin-bottom:12px;">
          <div style="display:flex; justify-content:space-between;">
            <span class="factor-label">{label}</span>
            <span style="font-size:0.7rem; color:#64748b">✅ {cnt_app}  ❌ {cnt_rej}</span>
          </div>
          <div style="display:flex; gap:2px;">
            <div style="flex:{w_app}; background:#22c55e44; border-radius:4px; height:8px;"></div>
            <div style="flex:{w_rej}; background:#ef444444; border-radius:4px; height:8px;"></div>
          </div>
        </div>"""
    st.markdown(f'<div class="card">{bars_html}<div style="font-size:0.72rem; color:#475569; margin-top:8px;">🟢 Approved &nbsp; 🔴 Rejected</div></div>', unsafe_allow_html=True)