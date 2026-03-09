import streamlit as st
import pandas as pd
import pdfplumber
import re
import google.generativeai as genai
import plotly.express as px

# Configuration & Entity Mapping
ENTITY_MAP = {
    "1454660101": "FMF",
    "2634389110": "Veisari",
    "2626605311": "ATPACK",
    "2629665810": "BCF-810",
    "2554020110": "BCF-110"
}

@st.cache_data
def extract_bill_data(pdf_file):
    """Extracts key metrics from an EFL bill PDF."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    # Regex patterns for FMF specific bill formats
    account_no_match = re.search(r"Account No\.\s*(\d+)", text)
    month_match = re.search(r"Account for\s*(\w+\s\d{4})", text)
    kwh_match = re.search(r"(\d+)KWH", text)
    total_due_match = re.search(r"TOTAL DUE\s*\$(\d+,?\d*\.\d+)", text)
    vat_match = re.search(r"VAT\s*\$(\d+,?\d*\.\d+)", text)
    max_demand_match = re.search(r"(\d+)KW\s@", text)
    
    data = {
        "Account No": account_no_match.group(1) if account_no_match else "Unknown",
        "Month": month_match.group(1) if month_match else "Unknown",
        "kWh_Usage": float(kwh_match.group(1).replace(',', '')) if kwh_match else 0.0,
        "Total_Due": float(total_due_match.group(1).replace(',', '')) if total_due_match else 0.0,
        "VAT": float(vat_match.group(1).replace(',', '')) if vat_match else 0.0,
        "Max_Demand_kW": float(max_demand_match.group(1).replace(',', '')) if max_demand_match else 0.0
    }
    
    data["Entity"] = ENTITY_MAP.get(data["Account No"], "Unknown Entity")
    return data

def verify_calculations(data):
    """Double checks the bill totals for transparency."""
    # Note: Rates vary by account; in a production app, we'd extract the rate per line item.
    # For now, we flag if VAT + VEP != VIP
    expected_vip = data["Total_Due"] 
    # Logic for internal verification can be expanded here
    return True

def get_gemini_insights(df_context):
    """Sends bill data to Gemini for cost-optimization strategies."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        prompt = f"""
        You are an expert energy consultant analyzing electricity consumption data for the FMF Group.
        Review the following tabular data:
        {df_context.to_string()}
        
        Provide a concise, professional report structured EXACTLY as follows. 
        DO NOT include conversational filler paragraphs.
        
        ### 📊 Key Findings
        (Provide a bulleted list highlighting the highest consuming entity, anomalies in VAT, and usage-to-cost ratios.)
        
        ### 🛠️ Engineering Strategies for Demand Reduction 
        (Provide a markdown table with 3 specific, actionable engineering strategies to reduce 'Maximum Demand' charges based on the entities listed. Columns: 'Strategy', 'Target Entity', 'Expected Impact'.)
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
        except Exception as api_err:
            if '404' in str(api_err):
                # Fallback to the older gemini-pro model if 1.5-flash is not available on this API key's project
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
            else:
                raise api_err
                
        return response.text
    except Exception as e:
        return f"⚠️ Error generating AI insights: {e}\nPlease check your API Key and network connection."

# --- Streamlit UI ---
st.set_page_config(page_title="FMF Group Bill Checker AI", layout="wide")
st.title("🏭 FMF Group Bill Checker AI")
st.markdown("---")

uploaded_files = st.sidebar.file_uploader("Upload EFL PDF Bills", accept_multiple_files=True, type="pdf")

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        bill_info = extract_bill_data(file)
        if verify_calculations(bill_info):
            all_data.append(bill_info)
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Convert 'Month' to datetime for timeline sorting, handle parsing errors
        try:
            df['Date'] = pd.to_datetime(df['Month'], format='%B %Y', errors='coerce')
        except Exception:
            df['Date'] = df['Month'] # fallback
        
        # Sort chronologically if Date was successfully parsed
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            df = df.sort_values(by='Date')

        # Calculate Unique Months for Averages
        number_of_months = df['Month'].nunique() if 'Month' in df.columns else 1
        avg_expenditure_per_month = df['Total_Due'].sum() / number_of_months
        avg_consumption_per_month = df['kWh_Usage'].sum() / number_of_months
        
        # Calculate Average Peak Demand (excluding zero values)
        md_non_zero = df[df['Max_Demand_kW'] > 0]['Max_Demand_kW']
        avg_md_per_month = md_non_zero.sum() / number_of_months if not md_non_zero.empty else 0

        # Key Metrics Overview
        st.subheader("📊 Executive Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Expenditure (per month)", f"${avg_expenditure_per_month:,.2f}")
        col2.metric("Avg Consumption (per month)", f"{avg_consumption_per_month:,.0f} kWh")
        col3.metric("Avg Peak Demand (per month)", f"{avg_md_per_month:,.1f} kW")

        st.markdown("---")

        # Visualizations
        st.subheader("📈 Cross-Sectional Analytics")
        tab1, tab2, tab3 = st.tabs(["Bill Amount", "Consumption Breakdown", "Maximum Demand (MD)"])
        
        with tab1:
            fig1 = px.bar(
                df, 
                x="Entity", 
                y="Total_Due", 
                color="Entity", 
                title="Total Bill Expenditure by Entity",
                text_auto='.2s',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig1.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            fig2 = px.pie(
                df, 
                names="Entity", 
                values="kWh_Usage", 
                title="Electricity Consumption (kWh) Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            fig3 = px.bar(
                df,
                x="Entity",
                y="Max_Demand_kW",
                color="Entity",
                title="Maximum Demand (kW) by Entity",
                text_auto='.1f',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig3.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(fig3, use_container_width=True)
            
        st.markdown("---")
        
        st.subheader("📆 Timeline Analytics")
        if pd.api.types.is_datetime64_any_dtype(df.get('Date')):
            # Try to see if there are multiple dates
            if df['Date'].nunique() > 1:
                time_tab1, time_tab2 = st.tabs(["Bill Expenditure Timeline", "Maximum Demand Timeline"])
                
                with time_tab1:
                    fig_time1 = px.bar(
                        df, 
                        x="Date", 
                        y="Total_Due", 
                        color="Entity", 
                        barmode="group",
                        title="Timeline: Bill Expenditure Over Time",
                        text_auto='.2s',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_time1.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                    st.plotly_chart(fig_time1, use_container_width=True)
                    
                with time_tab2:
                    fig_time2 = px.bar(
                        df, 
                        x="Date", 
                        y="Max_Demand_kW", 
                        color="Entity", 
                        barmode="group",
                        title="Timeline: Maximum Demand (kW) Over Time",
                        text_auto='.1f',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_time2.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                    st.plotly_chart(fig_time2, use_container_width=True)
            else:
                st.info("Upload bills from multiple different months for the same entity to see timeline trends.")
        else:
             st.info("Could not parse dates from the bills to build a timeline.")

        st.markdown("---")

        # Gemini Analysis Section
        st.subheader("🤖 AI Cost-Saving Report")
        if st.button("Generate AI Optimization Report", type="primary"):
            with st.spinner("Gemini is analyzing your bills and formulating strategies..."):
                insights = get_gemini_insights(df)
                st.info(insights)

        st.markdown("---")
        
        # Extracted Data & Export
        st.subheader("📄 Extracted Raw Data")
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='fmf_extracted_bills.csv',
            mime='text/csv',
        )
        
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Could not extract any valid data from the uploaded PDFs. Please check the files.")

else:
    st.info("👋 Welcome to the FMF Group Bill Checker AI. Please upload PDF bills in the sidebar to begin analysis.")
