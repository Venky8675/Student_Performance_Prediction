import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from pathlib import Path

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background & font */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; color: #e0e0e0; }

    /* Main header */
    .hero {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 36px 40px 28px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(102,126,234,0.4);
    }
    .hero h1 { color: #fff; font-size: 2.5rem; font-weight: 800; margin: 0; }
    .hero p  { color: #d9d9ff; font-size: 1.05rem; margin-top: 8px; }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform .2s;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-title { font-size: .85rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #a78bfa; margin-top: 6px; }
    .metric-sub   { font-size: .8rem; color: #888; margin-top: 4px; }

    /* Result banner */
    .result-pass {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 14px; padding: 24px 32px; color: #fff;
        font-size: 1.5rem; font-weight: 700; text-align: center;
        box-shadow: 0 6px 20px rgba(56,239,125,.3);
    }
    .result-fail {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        border-radius: 14px; padding: 24px 32px; color: #fff;
        font-size: 1.5rem; font-weight: 700; text-align: center;
        box-shadow: 0 6px 20px rgba(231,76,60,.3);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] .css-1d391kg { padding-top: 2rem; }

    /* Section headings */
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #c4b5fd;
        border-left: 4px solid #7c3aed; padding-left: 12px;
        margin-bottom: 16px;
    }

    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.07);
        border-radius: 8px 8px 0 0;
        color: #aaa;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,58,237,0.5) !important;
        color: #fff !important;
    }

    /* Gauge container */
    .gauge-wrap { display:flex; justify-content:center; margin: 10px 0; }

    /* Info box */
    .info-box {
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
        font-size: .9rem;
    }

    /* Progress bar override */
    .stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 10px;
        font-size: 1rem; font-weight: 600;
        padding: 10px 30px;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .85; }

    /* Slider label */
    .stSlider label { color: #c4b5fd !important; font-weight: 600; }

    /* Select box */
    .stSelectbox label { color: #c4b5fd !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "student_pass_fail_model.pkl"

FEATURE_ORDER = [
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "Gender",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities",
]

VALID_CATEGORIES = {
    "Gender":                   ["Female", "Male"],
    "Parental_Education_Level": ["Bachelors", "High School", "Masters", "None", "PhD"],
    "Internet_Access_at_Home":  ["No", "Yes"],
    "Extracurricular_Activities": ["No", "Yes"],
}

NUMERIC_RANGES = {
    "Study_Hours_per_Week": (10.0, 40.0),
    "Attendance_Rate":      (5.0,  100.0),
    "Past_Exam_Scores":     (40.0, 100.0),
}

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def predict(model, row: dict, threshold: float = 0.5):
    df_in = pd.DataFrame([row])[FEATURE_ORDER]
    classes = list(model.named_steps["model"].classes_)
    proba = model.predict_proba(df_in)[0]
    prob_pass = proba[classes.index(1)]
    prediction = "Pass" if prob_pass >= threshold else "Fail"
    return prediction, prob_pass

def gauge_html(prob: float) -> str:
    pct = prob * 100
    color = "#38ef7d" if prob >= 0.5 else "#e74c3c"
    angle = -90 + 180 * prob          # -90° → +90°
    return f"""
    <div style="display:flex;justify-content:center;margin:12px 0;">
      <div style="position:relative;width:200px;height:110px;overflow:hidden;">
        <!-- Background arc -->
        <svg width="200" height="110" viewBox="0 0 200 110">
          <path d="M 15 100 A 85 85 0 0 1 185 100" fill="none" stroke="#333" stroke-width="20" stroke-linecap="round"/>
          <path d="M 15 100 A 85 85 0 0 1 185 100" fill="none" stroke="{color}"
                stroke-width="20" stroke-linecap="round"
                stroke-dasharray="{pct * 2.67:.1f} 267"
                style="transform-origin:center;"/>
        </svg>
        <!-- Needle -->
        <div style="
          position:absolute; bottom:0; left:50%; width:4px; height:80px;
          background:{color}; border-radius:4px;
          transform-origin:bottom center;
          transform:translateX(-50%) rotate({angle:.0f}deg);
          box-shadow: 0 0 8px {color};
        "></div>
        <!-- Label -->
        <div style="
          position:absolute; bottom:4px; left:50%; transform:translateX(-50%);
          font-size:1.5rem; font-weight:800; color:{color};
        ">{pct:.1f}%</div>
      </div>
    </div>"""

def radar_chart(values: dict):
    """Mini radar for student profile."""
    labels = list(values.keys())
    stats  = list(values.values())
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    stats  += stats[:1]; angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True),
                           facecolor='none')
    ax.set_facecolor('none')
    ax.plot(angles, stats, 'o-', linewidth=2, color='#a78bfa')
    ax.fill(angles, stats, alpha=0.25, color='#7c3aed')
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, color='#c4b5fd', size=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([], color='none')
    ax.yaxis.grid(True, color=(1.0, 1.0, 1.0, 0.10))
    ax.xaxis.grid(True, color=(1.0, 1.0, 1.0, 0.15))
    ax.spines['polar'].set_color((1.0, 1.0, 1.0, 0.15))
    fig.patch.set_alpha(0)
    return fig

# ─── Load model ─────────────────────────────────────────────────────────────
model = load_model()

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎓 Student Performance Predictor</h1>
 
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        "⚠️ **Model file not found.**  "
        "Please run the notebook (`student_performance.ipynb`) first to generate "
        "`student_pass_fail_model.pkl`, then place it in the same folder as this app."
    )



# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Data Insights", "ℹ️ About"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">🧑‍🎓 Student Profile</div>', unsafe_allow_html=True)

        with st.expander("📚 Academic Metrics", expanded=True):
            study_hours = st.number_input(
                "Study Hours per Week",
                min_value=10.0, max_value=40.0, value=25.0, step=1.0,
                help="Type the average number of hours spent studying per week"
            )
            attendance = st.number_input(
                "Attendance Rate (%)",
                min_value=5.0, max_value=100.0, value=80.0, step=1.0,
                help="Type the percentage of classes attended"
            )
            past_scores = st.number_input(
                "Past Exam Scores",
                min_value=40.0, max_value=100.0, value=75.0, step=1.0,
                help="Type the average score from previous exams"
            )

        with st.expander("👤 Personal & Family", expanded=True):
            gender = st.selectbox("Gender", VALID_CATEGORIES["Gender"])
            parental_edu = st.selectbox(
                "Parental Education Level",
                VALID_CATEGORIES["Parental_Education_Level"]
            )
            internet = st.selectbox(
                "Internet Access at Home",
                VALID_CATEGORIES["Internet_Access_at_Home"]
            )
            extracurricular = st.selectbox(
                "Extracurricular Activities",
                VALID_CATEGORIES["Extracurricular_Activities"]
            )

     

        predict_btn = st.button("🔮 Predict Outcome", use_container_width=True)

    # ── Result panel ────────────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            if model is None:
                st.error("Model not loaded. See the error message above.")
            else:
                row = {
                    "Gender": gender,
                    "Study_Hours_per_Week": float(study_hours),
                    "Attendance_Rate":      float(attendance),
                    "Past_Exam_Scores":     float(past_scores),
                    "Parental_Education_Level":    parental_edu,
                    "Internet_Access_at_Home":     internet,
                    "Extracurricular_Activities":  extracurricular,
                }
                prediction, prob_pass = predict(model, row)
                prob_fail = 1 - prob_pass

                css_cls = "result-pass" if prediction == "Pass" else "result-fail"
                emoji    = "✅" if prediction == "Pass" else "❌"
                st.markdown(
                    f'<div class="{css_cls}">{emoji} Predicted: <b>{prediction}</b></div>',
                    unsafe_allow_html=True
                )

                # Gauge
                st.markdown(gauge_html(prob_pass), unsafe_allow_html=True)

                # Probabilities
                col_p, col_f = st.columns(2)
                col_p.metric("P(Pass)", f"{prob_pass*100:.1f}%")
                col_f.metric("P(Fail)", f"{prob_fail*100:.1f}%")

                st.progress(prob_pass)

                # Risk level
                if prob_pass >= 0.75:
                    risk_label, risk_color = "Low Risk 🟢", "#22c55e"
                elif prob_pass >= 0.50:
                    risk_label, risk_color = "Moderate Risk 🟡", "#eab308"
                else:
                    risk_label, risk_color = "High Risk 🔴", "#ef4444"

                st.markdown(
                    f'<div class="info-box">Risk Level: '
                    f'<b style="color:{risk_color}">{risk_label}</b></div>',
                    unsafe_allow_html=True
                )

                
                # Actionable advice
                st.markdown("---")
                st.markdown("**💡 Personalised Advice**")
                tips = []
                if study_hours < 20:
                    tips.append("📖 Increase weekly study hours — aim for at least 25 h/week.")
                if attendance < 75:
                    tips.append("🏫 Improve class attendance — below 75% is a strong predictor of failure.")
                if past_scores < 65:
                    tips.append("📝 Focus on past-exam weak areas; seek tutoring or extra resources.")
                if internet == "No":
                    tips.append("🌐 Access to online resources can significantly support learning.")
                if extracurricular == "No":
                    tips.append("🏅 Extracurricular activities can improve motivation and study habits.")
                if not tips:
                    tips = ["🌟 Excellent profile! Keep maintaining these habits."]
                for t in tips:
                    st.write(t)

        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:40px;">
              <span style="font-size:3rem">🔮</span><br><br>
              Fill in the student profile on the left<br>and click <b>Predict Outcome</b>
              to see the result.
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – DATA INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📈 Visualise the Data</div>', unsafe_allow_html=True)

    # Try to load dataset
    data_path = Path(__file__).parent / "student_performance_data.csv"
    cleaned_path = Path(__file__).parent / "student_performance_cleaned.csv"

    df_raw = None
    if data_path.exists():
        df_raw = pd.read_csv(data_path)
    elif cleaned_path.exists():
        df_raw = pd.read_csv(cleaned_path)

    if df_raw is None:
        st.warning(
            "⚠️ Dataset CSV not found. Place `student_performance_data.csv` "
            "in the same folder as this app to enable interactive charts."
        )
        st.markdown("""
        <div class="info-box">
        <b>Dataset summary (from notebook):</b><br>
        • 10,000 students · 10 columns<br>
        • 4 numeric features, 4 categorical features<br>
        • Target: Pass / Fail (binary classification)<br>
        • ~8% missing values (imputed during preprocessing)<br>
        • ~52% Pass, ~48% Fail (balanced)
        </div>""", unsafe_allow_html=True)

        # Static model comparison chart from notebook results
        st.markdown('<div class="section-title">🏆 Model Comparison</div>', unsafe_allow_html=True)
        models = ["Dummy\nBaseline", "Decision\nTree", "Naive\nBayes",
                  "KNN", "Gradient\nBoosting", "SVM", "Logistic\nRegression", "Random\nForest"]
        accs   = [52.0, 71.0, 77.0, 85.0, 86.0, 87.0, 87.5, 87.95]

        fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
        ax.set_facecolor('none')
        colors = ['#555']*7 + ['#7c3aed']
        bars = ax.bar(models, accs, color=colors, edgecolor='none', width=0.6, zorder=3)
        ax.bar_label(bars, fmt='%.1f%%', color='white', fontsize=9, padding=3)
        ax.set_ylim(0, 102)
        ax.set_ylabel("Test Accuracy (%)", color='#aaa')
        ax.tick_params(colors='#aaa')
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.15))
        ax.yaxis.grid(True, color=(1.0, 1.0, 1.0, 0.08), zorder=0)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Feature importance (from notebook)
        st.markdown('<div class="section-title">🔑 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        feat_names = ["Past_Exam_Scores", "Attendance_Rate", "Study_Hours_per_Week",
                      "Parental_Education_Level", "Gender",
                      "Internet_Access_at_Home", "Extracurricular_Activities"]
        importances = [0.1558, 0.1530, 0.1490, 0.1310, 0.09, 0.08, 0.075]

        fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor='none')
        ax2.set_facecolor('none')
        clrs = ['#7c3aed' if i < 3 else '#555' for i in range(len(feat_names))]
        bars2 = ax2.barh(feat_names[::-1], importances[::-1], color=clrs[::-1], edgecolor='none')
        ax2.bar_label(bars2, fmt='%.3f', color='white', fontsize=9, padding=3)
        ax2.tick_params(colors='#aaa')
        ax2.spines[['top','right','bottom']].set_visible(False)
        ax2.spines['left'].set_color((1.0, 1.0, 1.0, 0.15))
        ax2.xaxis.grid(True, color=(1.0, 1.0, 1.0, 0.08), zorder=0)
        fig2.patch.set_alpha(0)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    else:
        # Interactive EDA with real data
        st.write(f"**Dataset:** {len(df_raw):,} rows × {df_raw.shape[1]} columns")

        viz_choice = st.selectbox(
            "Choose a visualisation",
            ["Pass/Fail Distribution", "Study Hours vs Attendance",
             "Feature Distributions", "Correlation Heatmap",
             "Model Comparison (from notebook)"]
        )

        fig_w, ax_w = plt.subplots(figsize=(10, 5), facecolor='none')
        ax_w.set_facecolor('none')

        if "Pass_Fail" in df_raw.columns:
            pf_col = "Pass_Fail"
        else:
            pf_col = None

        if viz_choice == "Pass/Fail Distribution" and pf_col:
            counts = df_raw[pf_col].value_counts()
            ax_w.bar(counts.index, counts.values,
                     color=['#38ef7d', '#e74c3c'], edgecolor='none', width=0.5)
            ax_w.set_ylabel("Count", color='#aaa'); ax_w.set_title("Pass vs Fail", color='white')

        elif viz_choice == "Study Hours vs Attendance":
            sc_colors = df_raw[pf_col].map({"Pass": "#38ef7d", "Fail": "#e74c3c"}) if pf_col else '#a78bfa'
            ax_w.scatter(df_raw["Study_Hours_per_Week"], df_raw["Attendance_Rate"],
                         c=sc_colors, alpha=0.3, s=10)
            ax_w.set_xlabel("Study Hours/Week", color='#aaa')
            ax_w.set_ylabel("Attendance Rate (%)", color='#aaa')
            ax_w.set_title("Study Hours vs Attendance", color='white')
            if pf_col:
                legend_els = [mpatches.Patch(color='#38ef7d', label='Pass'),
                               mpatches.Patch(color='#e74c3c', label='Fail')]
                ax_w.legend(handles=legend_els, facecolor='#1e1b2e', labelcolor='white')

        elif viz_choice == "Feature Distributions":
            feat = st.selectbox("Select feature",
                                ["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores"])
            if pf_col:
                for outcome, col in [("Pass", "#38ef7d"), ("Fail", "#e74c3c")]:
                    subset = df_raw[df_raw[pf_col] == outcome][feat].dropna()
                    ax_w.hist(subset, bins=30, alpha=0.6, color=col, label=outcome)
                ax_w.legend(facecolor='#1e1b2e', labelcolor='white')
            else:
                ax_w.hist(df_raw[feat].dropna(), bins=30, color='#7c3aed', alpha=0.8)
            ax_w.set_xlabel(feat, color='#aaa'); ax_w.set_title(f"Distribution of {feat}", color='white')

        elif viz_choice == "Correlation Heatmap":
            num_df = df_raw.select_dtypes(include="number")
            corr   = num_df.corr()
            plt.close()
            fig_w, ax_w = plt.subplots(figsize=(8, 6), facecolor='none')
            ax_w.set_facecolor('none')
            sns.heatmap(corr, ax=ax_w, cmap='RdYlGn', annot=True, fmt='.2f',
                        linewidths=0.5, square=True,
                        cbar_kws={"shrink": 0.8})
            ax_w.set_title("Feature Correlation Heatmap", color='white')
            ax_w.tick_params(colors='#aaa')

        elif viz_choice == "Model Comparison (from notebook)":
            models = ["Dummy\nBaseline", "Decision\nTree", "Naive\nBayes",
                      "KNN", "Gradient\nBoosting", "SVM", "Logistic\nRegression", "Random\nForest"]
            accs   = [52.0, 71.0, 77.0, 85.0, 86.0, 87.0, 87.5, 87.95]
            colors_m = ['#555']*7 + ['#7c3aed']
            bars = ax_w.bar(models, accs, color=colors_m, edgecolor='none', width=0.6, zorder=3)
            ax_w.bar_label(bars, fmt='%.1f%%', color='white', fontsize=9, padding=3)
            ax_w.set_ylim(0, 102)
            ax_w.set_title("Model Comparison – Test Accuracy", color='white')
            ax_w.set_ylabel("Accuracy (%)", color='#aaa')

        ax_w.tick_params(colors='#aaa')
        for spine in ax_w.spines.values():
            spine.set_color((1.0, 1.0, 1.0, 0.15))
        fig_w.patch.set_alpha(0)
        st.pyplot(fig_w, use_container_width=True)
        plt.close()

        # Confusion matrix static display
        st.markdown('<div class="section-title">🧮 Final Model Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array([[900, 138], [103, 859]])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor='none')
        ax_cm.set_facecolor('none')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax_cm,
                    xticklabels=['Pred Fail','Pred Pass'],
                    yticklabels=['Actual Fail','Actual Pass'])
        ax_cm.set_title("Confusion Matrix (Test Set, n=2000)", color='white')
        ax_cm.tick_params(colors='#aaa')
        fig_cm.patch.set_alpha(0)
        st.pyplot(fig_cm, use_container_width=False)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">ℹ️ About this Project</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
        <b>🎯 Objective</b><br>
        Predict whether a student will <b>Pass</b> or <b>Fail</b> based on academic
        and demographic features, using classical machine-learning classification.
        </div>

        <div class="info-box">
        <b>📦 Dataset</b><br>
        • 10,000 students<br>
        • 10 columns (3 numeric predictors, 4 categorical predictors, 1 target, 1 ID, 1 leaky column excluded)<br>
        • Source: <code>student_performance_data.csv</code>
        </div>

        <div class="info-box">
        <b>🔬 Workflow</b><br>
        Day 1–2: Data loading, EDA, missing-value imputation, cleaning<br>
        Day 3: Visualisation, distribution analysis, correlation heatmap<br>
        Day 4–5: Encoding, train/test split (80/20), pipeline building,
        7 models evaluated + cross-validation<br>
        Day 5: Final model selection (Random Forest), threshold analysis, feature importance
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="info-box">
        <b>🏆 Final Model – Random Forest</b><br>
        • n_estimators = 100, random_state = 0<br>
        • 5-fold Stratified CV (training data only)<br>
        • Test Accuracy: <b>87.95%</b><br>
        • Fail Recall: <b>86.32%</b><br>
        • Macro F1: <b>88.00%</b>
        </div>

        <div class="info-box">
        <b>🔑 Top Features (Permutation Importance)</b><br>
        1. Past Exam Scores (0.156)<br>
        2. Attendance Rate (0.153)<br>
        3. Study Hours / Week (0.149)<br>
        4. Parental Education Level (0.131)<br>
        5. Gender (0.09)
        </div>

        <div class="info-box">
        <b>⚙️ Tech Stack</b><br>
        Python · scikit-learn · pandas · numpy · matplotlib · seaborn · joblib · Streamlit
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ---
    <div style="text-align:center;color:#555;font-size:.85rem;">
    Student Performance Predictor · Built with Streamlit · Internship Assignment Project
    </div>""", unsafe_allow_html=True)
