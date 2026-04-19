import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
import altair as alt
import os
from dotenv import load_dotenv

load_dotenv()

from src.predict import predict
from src.agent import build_agent
from src.pdf_export import generate_pdf_report

# Initialize required session states
if "df" not in st.session_state:
    st.session_state["df"] = None
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "report" not in st.session_state:
    st.session_state["report"] = None
if "agent_ran" not in st.session_state:
    st.session_state["agent_ran"] = False

# Page Configuration
st.set_page_config(page_title="Clinical No-Show Predictor", layout="wide", page_icon="🏥")

@st.cache_resource
def get_model():
    return joblib.load("models/noshow_model.pkl")

@st.cache_resource
def get_agent():
    return build_agent()

@st.cache_data
def get_predictions(df: pd.DataFrame):
    return predict(df)

model = get_model()
agent = get_agent()

# Application Title
st.title("🏥 Clinical Appointment No-Show Prediction System")

# Create Tabs
tab1, tab2 = st.tabs(["📊 ML Predictions", "🤖 AI Care Coordinator"])

with tab1:
    st.header("Upload Data for Risk Prediction")

    # Sidebar for Sample Data Download
    with st.sidebar:
        st.header("Sample Data for Testing")
        st.write("Download the sample dataset below to test out the prediction system.")
        try:
            sample_df = pd.read_csv("sample_data.csv")
            csv_data = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Sample CSV",
                data=csv_data,
                file_name="sample_data.csv",
                mime="text/csv",
            )
            st.subheader("Sample Data Preview")
            st.dataframe(sample_df)
        except Exception as e:
            st.error(f"Sample data file not found or error loading: {e}")

    # File Uploader
    uploaded_file = st.file_uploader("Upload Appointment CSV", type=["csv"], key="uploader_tab1")

    if uploaded_file:
        try:
            # Load Data
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df

            # Predict Probabilities
            probs = get_predictions(df)

            # Store to state and dataframe
            df["No_show_risk_probability"] = probs
            df["Risk_Level"] = pd.cut(
                df["No_show_risk_probability"],
                bins=[0, 0.3, 0.6, 1],
                labels=["Low", "Medium", "High"],
                include_lowest=True
            )
            st.session_state["predictions"] = df

            # Overview Dashboard
            st.subheader("Overview Dashboard")
            risk_counts = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Patients", len(df))
            col2.metric("High Risk", int(risk_counts.get("High", 0)), delta_color="inverse")
            col3.metric("Medium Risk", int(risk_counts.get("Medium", 0)), delta_color="off")
            col4.metric("Low Risk", int(risk_counts.get("Low", 0)), delta_color="normal")
            
            st.markdown("---")

            # Prediction Output Display
            st.subheader("Patient Risk Table")
            st.caption("Detailed view of patient risk probabilities and categorization. Sorted by highest risk first.")
            
            # Select relevant columns for display
            display_cols = []
            for col in ["PatientId", "AppointmentID", "Age", "Gender", "Neighbourhood"]:
                if col in df.columns:
                    display_cols.append(col)
                    
            display_cols.extend(["No_show_risk_probability", "Risk_Level"])
            
            # Sort by highest risk
            display_df = df[display_cols].sort_values(by="No_show_risk_probability", ascending=False)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                column_config={
                    "PatientId": st.column_config.TextColumn("Patient ID"),
                    "AppointmentID": st.column_config.TextColumn("Appt ID"),
                    "Age": st.column_config.NumberColumn("Age", format="%d"),
                    "Gender": st.column_config.TextColumn("Gender"),
                    "Neighbourhood": st.column_config.TextColumn("Neighbourhood"),
                    "No_show_risk_probability": st.column_config.ProgressColumn(
                        "Risk Probability",
                        help="The predicted probability of the patient not showing up.",
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    "Risk_Level": st.column_config.TextColumn(
                        "Risk Level",
                        help="Categorized as Low (<30%), Medium (30-60%), or High (>60%)"
                    )
                },
                hide_index=True,
            )

            st.markdown("---")

            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Risk Distribution")
                risk_df = risk_counts.reset_index()
                risk_df.columns = ["Risk Level", "Count"]
                
                chart = alt.Chart(risk_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("Risk Level", sort=["Low", "Medium", "High"], title="Risk Level"),
                    y=alt.Y("Count", title="Number of Patients"),
                    color=alt.Color("Risk Level", scale=alt.Scale(
                        domain=["Low", "Medium", "High"],
                        range=["#28a745", "#ffc107", "#dc3545"]
                    ), legend=None),
                    tooltip=["Risk Level", "Count"]
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)

            with col_chart2:
                st.subheader("Key Risk Drivers")
                importance = model.feature_importances_
                feature_names = model.feature_names_in_
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False).head(10)
                
                driver_chart = alt.Chart(importance_df).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
                    x=alt.X("Importance", title="Relative Importance"),
                    y=alt.Y("Feature", sort="-x", title=""),
                    color=alt.value("#4C72B0"),
                    tooltip=["Feature", "Importance"]
                ).properties(height=350)
                st.altair_chart(driver_chart, use_container_width=True)

            st.success("Prediction completed successfully! You can now use the AI Care Coordinator tab.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab2:
    st.header("AI Care Coordinator")
    
    if st.session_state["predictions"] is None:
        st.warning("Please upload a CSV file and run predictions in the 'ML Predictions' tab first.")
    else:
        # Load the predicted data from state
        df = st.session_state["predictions"]
        
        # Calculate Context to feed to Agent
        high_risk_df = df[df["Risk_Level"] == "High"]
        total_high_risk = len(high_risk_df)
        
        st.info(f"**High-Risk Patients Identified**: {total_high_risk}")
        
        if total_high_risk > 0:
            avg_prob = float(high_risk_df["No_show_risk_probability"].mean())
            
            # Very basic risk factors (custom logic for context)
            factors = []
            if "LeadTime" in high_risk_df.columns:
                factors.append("extended lead times")
            if "SMS_received" in high_risk_df.columns and high_risk_df["SMS_received"].mean() < 0.5:
                factors.append("low SMS verification rates")
            if "Age" in high_risk_df.columns:
                factors.append(f"average age of {int(high_risk_df['Age'].mean())}")
                
            patient_data = {
                "total_high_risk": total_high_risk,
                "avg_probability": avg_prob,
                "common_factors": factors if factors else ["general demographic correlations"]
            }

            # Generate Report Button
            if st.button("Generate Care Coordination Report"):
                with st.spinner("Generating evidence-based guidelines and report..."):
                    # Check for groq key
                    api_key = os.environ.get("GROQ_API_KEY")
                    if not api_key:
                        try:
                            api_key = st.secrets["GROQ_API_KEY"]
                        except:
                            st.warning("Groq API key not configured. Add GROQ_API_KEY to .env or Streamlit secrets.")
                            st.stop()
                        
                    try:
                        # Invoke Agent
                        state = {
                            "patient_data": patient_data,
                            "risk_summary": "",
                            "retrieved_guidelines": "",
                            "final_report": {},
                            "error": ""
                        }
                        
                        final_state = agent.invoke(state)
                        
                        if final_state.get("error"):
                            st.error(f"Agent execution encountered an error: {final_state['error']}")
                        else:
                            st.session_state["report"] = final_state.get("final_report", {})
                            st.session_state["agent_ran"] = True
                            st.success("Care Coordination Report Generated")
                            
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")

            # Display the Generated Report structurally
            if st.session_state["agent_ran"] and st.session_state["report"]:
                report = st.session_state["report"]
                
                # 1. Risk Summary
                st.subheader("📋 Risk Summary")
                st.info(report.get("risk_summary", "No summary provided."))
                
                # 2. Key Contributing Factors
                st.subheader("⚠️ Key Contributing Factors")
                factors = report.get("contributing_factors", [])
                for factor in factors:
                    st.markdown(f"- {factor}")
                    
                # 3. Intervention Strategies
                st.subheader("🎯 Intervention Strategies")
                strategies = report.get("intervention_strategies", [])
                for strat in strategies:
                    priority = strat.get("priority", "Medium")
                    # Badge color coding
                    badge = "🟡"
                    if "High" in priority: badge = "🔴"
                    elif "Low" in priority: badge = "🟢"
                    
                    with st.expander(f"{badge} {strat.get('strategy', 'Strategy')} ({priority} Priority)"):
                        st.write(strat.get("description", "No description provided."))

                # 4. Sources
                st.subheader("📚 Sources & References")
                sources = report.get("sources", [])
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {source}")
                    
                # 5. Disclaimers
                st.subheader("⚖️ Disclaimers")
                st.warning(f"**Operational Disclaimer:** {report.get('operational_disclaimer', '')}")
                st.warning(f"**Ethical Disclaimer:** {report.get('ethical_disclaimer', '')}")
                
                # PDF Download button
                pdf_bytes = generate_pdf_report(report)
                
                st.download_button(
                    label="📄 Download Report as PDF",
                    data=pdf_bytes,
                    file_name="care_coordination_report.pdf",
                    mime="application/pdf"
                )
        else:
            st.success("No high-risk patients were identified in the current upload.")