import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
from model import extract_text_from_pdf, extract_customer_requirements, extract_company_policies, extract_customer_objections

# Set page config with a wide layout and custom theme
st.set_page_config(page_title="Car Sales Conversation Analyzer", layout="wide", page_icon="🚗")

# Apply custom CSS for better styling
st.markdown("""
    <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #f7f9fc;
        }
        .block-container {
            padding-top: 2rem;
        }
        .css-18ni7ap {
            display: flex;
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and subtitle
st.title("🚗 Car Sales Conversation Information Extractor and Analyzer")
st.subheader("Analyze customer conversations and extract insights to improve your sales strategy.")

# Sidebar instructions
st.sidebar.title("📋 Instructions")
st.sidebar.write("""
1. Upload conversation transcripts in PDF format.
2. Click 'Analyze' to extract key information.
3. View extracted data, visualizations, and download reports.
""")

# File uploader for single and bulk uploads
uploaded_files = st.file_uploader("Upload Conversation Transcripts (PDF)", type=["pdf"], accept_multiple_files=True)

# Progress bar and status message
progress_bar = st.progress(0)
status_text = st.empty()

def export_as_pdf(dataframe):
    """Export the dataframe as a PDF file."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=1, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for row in dataframe.values:
        row_text = ', '.join(map(str, row))
        pdf.multi_cell(0, 10, row_text)
    return pdf.output(dest='S').encode('latin1')

# Button to process files
if st.button("Analyze 🕵️‍♂️"):
    if uploaded_files:
        all_results = []
        for i, file in enumerate(uploaded_files):
            # Update progress bar and status
            progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text(f"Processing {file.name}...")

            # Extract text from each uploaded file
            text = extract_text_from_pdf(file)

            # Extract information
            requirements = extract_customer_requirements(text)
            policies = extract_company_policies(text)
            objections = extract_customer_objections(text)

            # Compile results into a JSON-like format
            result = {
                "filename": file.name,
                "customer_requirements": requirements,
                "company_policies": policies,
                "customer_objections": objections
            }
            all_results.append(result)

        # Clear progress and status
        progress_bar.empty()
        status_text.empty()

        # Display results in JSON format
        st.subheader("📝 Extracted Information (JSON Format)")
        st.json(all_results)

        # Convert results to a DataFrame
        results_df = pd.json_normalize(all_results)
        st.subheader("📊 Extracted Information Table")
        st.dataframe(results_df)

        # Advanced Visualizations
        st.subheader("📈 Advanced Analysis Dashboard")
        
        # Distribution of Car Types
        if 'customer_requirements.car_type' in results_df:
            car_types = results_df['customer_requirements.car_type'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=car_types.index, y=car_types.values, palette="Blues_d", ax=ax)
            ax.set_title('Distribution of Car Types')
            st.pyplot(fig)

        # Distribution of Car Colors
        if 'customer_requirements.color' in results_df:
            car_colors = results_df['customer_requirements.color'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=car_colors.index, y=car_colors.values, palette="Greens_d", ax=ax)
            ax.set_title('Distribution of Car Colors')
            st.pyplot(fig)

        # Customer Objections Frequency
        if 'customer_objections' in results_df:
            objections = pd.json_normalize(results_df['customer_objections']).apply(pd.Series).stack().value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=objections.index, y=objections.values, palette="Reds_d", ax=ax)
            ax.set_title('Customer Objections Frequency')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)

        # Correlation Heatmap for Customer Requirements
        st.subheader("📊 Correlation Analysis of Customer Preferences")
        if 'customer_requirements' in results_df:
            # Example data processing for correlation, adjust fields based on actual data structure
            correlation_data = results_df[['customer_requirements.car_type', 'customer_requirements.fuel_type']]
            correlation_data = pd.get_dummies(correlation_data, drop_first=True)
            corr = correlation_data.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title('Correlation Matrix of Customer Requirements')
            st.pyplot(fig)

        # Export options
        st.subheader("📥 Export Analysis")
        export_format = st.selectbox("Select Export Format", ["CSV", "PDF"])
        
        if export_format == "CSV":
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV 📄",
                data=csv,
                file_name='car_sales_analysis.csv',
                mime='text/csv',
            )
        elif export_format == "PDF":
            pdf_data = export_as_pdf(results_df)
            st.download_button(
                label="Download PDF 📄",
                data=pdf_data,
                file_name='car_sales_analysis.pdf',
                mime='application/pdf',
            )
    else:
        st.warning("⚠️ Please upload at least one PDF file to analyze.")
