import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# === Streamlit Page Setup ===
st.set_page_config(
    page_title="🕵️ Data Detective Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Header ===
st.title("🕵️ Data Detective Pro: AI-Powered Pattern & Anomaly Explorer")
st.markdown("""
Welcome to **Data Detective Pro** – your one-stop solution for intelligent data exploration.  
Simply upload a `.csv` file and unlock:
- 📌 Anomaly Detection
- 🔗 Correlation Heatmaps
- 🧩 Pattern Discovery
- 🤖 Smart AI Assistant for Quick Q&A
""")

# === File Upload ===
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

@st.cache_data(show_spinner=False)
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df.copy()  # Return a copy to avoid cache warnings
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        # === Data Preview ===
        with st.expander("🔍 Data Preview", expanded=True):
            st.success(f"✅ Loaded {uploaded_file.name} | Shape: {df.shape}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 First 5 Rows")
                st.dataframe(df.head())
            with col2:
                st.subheader("📊 Summary Statistics")
                st.dataframe(df.describe())

        if not num_cols:
            st.warning("⚠️ No numeric columns detected. Please upload a dataset with numerical features for full functionality.")
        else:
            # === Anomaly Detection ===
            st.subheader("🔴 AI-Powered Anomaly Detection")
            contamination = st.slider("Outlier Sensitivity", 0.01, 0.2, 0.05, help="Higher = more points flagged as outliers")

            clf = IsolationForest(contamination=contamination, random_state=42)
            clf.fit(df[num_cols])
            df['anomaly_score'] = clf.decision_function(df[num_cols])
            df['anomaly'] = clf.predict(df[num_cols])

            anomalies = df[df['anomaly'] == -1]
            st.write(f"📌 **Detected Anomalies:** {len(anomalies)} records ({len(anomalies)/len(df):.2%})")

            with st.expander("📉 View Anomalies"):
                st.dataframe(anomalies.sort_values("anomaly_score", ascending=True))
                st.download_button("💾 Download Anomalies", anomalies.to_csv(index=False), "anomalies.csv")

            # === Correlation Analysis ===
            st.subheader("📈 Correlation Heatmap")
            corr_method = st.selectbox("Choose correlation method", ["pearson", "spearman", "kendall"])

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(method=corr_method),
                        annot=True, cmap='coolwarm', center=0, ax=ax, fmt=".2f")
            st.pyplot(fig)

            # === Pattern Discovery ===
            st.subheader("🧩 Pattern Discovery")
            corr_matrix = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
            top_corrs = corr_matrix[corr_matrix < 1].drop_duplicates().head(5)

            st.markdown("### 🔝 Top 5 Feature Relationships")
            for (pair, value) in top_corrs.items():
                st.markdown(f"- **{pair[0]}** ↔ **{pair[1]}** → Correlation: `{value:.2f}`")

            # === AI Assistant ===
            st.subheader("🤖 AI Analyst Assistant")
            question = st.text_input("Ask about your data (e.g. 'What are the top anomalies?' or 'Show feature relationships')")

            if question:
                q_lower = question.lower()
                if any(key in q_lower for key in ["correlat", "relat", "connect"]):
                    pair = top_corrs.index[0]
                    st.success(f"🔍 Strongest relationship: **{pair[0]}** & **{pair[1]}** (Correlation: `{top_corrs[0]:.2f}`)")
                elif any(key in q_lower for key in ["outlier", "anomal"]):
                    if not anomalies.empty:
                        top_anomaly = anomalies.sort_values("anomaly_score").iloc[0]
                        st.success(f"🚨 Top Anomaly at index `{top_anomaly.name}` | Score: `{top_anomaly.anomaly_score:.3f}`")
                        st.dataframe(top_anomaly.to_frame().T)
                    else:
                        st.info("✅ No significant anomalies detected.")
                else:
                    st.info("Try asking about:\n- 'top correlations'\n- 'show anomalies'\n- 'pattern between x and y'")

    except Exception as e:
        st.error(f"🚫 Error processing file: `{e}`")

else:
    st.info("📂 Please upload a CSV file to begin analysis.")
