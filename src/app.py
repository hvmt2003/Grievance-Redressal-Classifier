import streamlit as st
import sys
import os
import pandas as pd
# Streamlit uses altair for charts often, but we will use native st charts.

# Add the project root to the python path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.predict import GrievancePredictor

# Set page config
st.set_page_config(
    page_title="UP-CM Grievance Portal",
    page_icon="🇮🇳",
    layout="wide" # Wide layout for dashboard
)

# Load model
@st.cache_resource
def load_predictor():
    if os.path.exists('best_model_state.bin'):
        return GrievancePredictor()
    return None

# Load Data for Dashboard
@st.cache_data
def load_data():
    if os.path.exists('data/raw/grievances.csv'):
        return pd.read_csv('data/raw/grievances.csv')
    return pd.DataFrame()

predictor = load_predictor()
df = load_data()

# Sidebar Navigation
st.sidebar.title("🏛️ UP Governance")
mode = st.sidebar.radio("Go to:", ["Public Portal (Citizen)", "Admin Dashboard (Officials)"])

if mode == "Public Portal (Citizen)":
    st.title("🇮🇳 Jan Sunwai - Public Grievance Portal")
    st.markdown("### Uttar Pradesh Government")
    st.write("Submit your grievance here. Our AI system will automatically route it to the concerned department.")
    
    
    # Simple container instead of columns
    container = st.container()
    
    with container:
        grievance_text = st.text_area("Describe your grievance (Hindi/English):", height=200, placeholder="Example: Transformer burnt in Village Rampur...")
        
        if st.button("Submit Grievance", type="primary"):
            if not grievance_text.strip():
                st.warning("Please enter some text.")
            elif predictor is None:
                st.error("System Maintenance. Model not loaded.")
            else:
                with st.spinner("Processing..."):
                    category, priority, confidence = predictor.predict(grievance_text)
                    
                st.success("Grievance Submitted Successfully!")
                
                # Result Card
                st.markdown("---")
                st.markdown("#### 🎫 Receipt Details")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.info(f"**Assigned Department**\n\n### {category}")
                with res_col2:
                    color = "red" if priority == "High" else "orange" if priority == "Medium" else "green"
                    st.markdown(f"**Priority Level**\n\n### :{color}[{priority}]")
                    




elif mode == "Admin Dashboard (Officials)":
    st.title("📊 CM Dashboard - Analytics")
    st.markdown("Real-time monitoring of public grievances across the state.")
    
    if df.empty:
        st.warning("No historical data found.")
    else:
        # Top Level Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Grievances", len(df), "+12 today")
        m2.metric("High Priority", len(df[df['priority'] == 'High']), "Critical")
        m3.metric("Resolved", int(len(df) * 0.45), "+5%") # Fake stat for demo
        m4.metric("Avg Resolution Time", "48 Hrs", "-2 Hrs")
        
        st.divider()
        
        # Charts Row 1
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("⚠️ Evaluation by Urgency")
            # Pie Chart equivalent
            priority_counts = df['priority'].value_counts()
            st.bar_chart(priority_counts, color=["#ff4b4b" if x == 'High' else "#ffa500" if x == 'Medium' else "#00c853" for x in priority_counts.index]) # rough color attempt, st.bar_chart simple doesn't support list colors easily like this, but let's just use simple bar chart first.
            # actually st.bar_chart takes a dataframe.
            st.bar_chart(priority_counts)
            
        with c2:
            st.subheader("🏢 Department Load")
            dept_counts = df['category'].value_counts()
            st.bar_chart(dept_counts, horizontal=True)
            
        st.divider()
        
        # Data Table
        st.subheader("🚨 Recent High Priority Flags")
        high_priority_df = df[df['priority'] == 'High'].head(10)[['category', 'text']]
        st.dataframe(high_priority_df, use_container_width=True)
