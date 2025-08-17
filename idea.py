import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import io
import datetime # For timestamp in report

# --- New Import for PDF Generation ---
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    st.warning("WeasyPrint not found. PDF report generation will be disabled. Please install it using 'pip install weasyprint'.")
    WEASYPRINT_AVAILABLE = False
# --- End New Import ---

st.set_page_config(page_title="AI-Powered Survey Processor", layout="wide")
st.title("🧠 AI-Powered Survey Data Cleaning & Reporting Tool")

# Function to generate PDF report
def generate_pdf_report(cleaned_df, selected_cols, impute_method, apply_rules, initial_missing_report, outlier_info, rule_applied):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Survey Data Cleaning Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 20mm; }}
            h1 {{ color: #333; text-align: center; }}
            h2 {{ color: #555; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .section {{ margin-bottom: 15px; }}
            .note {{ font-style: italic; color: #777; }}
        </style>
    </head>
    <body>
        <h1>Survey Data Cleaning Report</h1>
        <p class="note">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="section">
            <h2>1. Overview</h2>
            <ul>
                <li><strong>Original Data Dimensions:</strong> {initial_shape[0]} rows, {initial_shape[1]} columns</li>
                <li><strong>Cleaned Data Dimensions:</strong> {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns</li>
                <li><strong>Selected Numeric Columns for Cleaning:</strong> {', '.join(selected_cols) if selected_cols else 'None'}</li>
            </ul>
        </div>

        <div class="section">
            <h2>2. Missing Value Imputation</h2>
            <p><strong>Method Used:</strong> {impute_method}</p>
    """
    if not initial_missing_report.empty:
        html_content += "<p><strong>Initial Missing Values (per selected column):</strong></p><ul>"
        for col, count in initial_missing_report.items():
            html_content += f"<li>{col}: {int(count)} values</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No initial missing values detected in selected columns.</p>"

    missing_after_cleaning = cleaned_df[selected_cols].isnull().sum()
    missing_after_cleaning = missing_after_cleaning[missing_after_cleaning > 0]
    if not missing_after_cleaning.empty:
        html_content += "<p><strong>Missing Values After Imputation/Cleaning (per selected column):</strong></p><ul>"
        for col, count in missing_after_cleaning.items():
            html_content += f"<li>{col}: {int(count)} values (may be due to rule-based checks or outlier removal)</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No missing values in selected columns after imputation and cleaning.</p>"

    html_content += f"""
        </div>

        <div class="section">
            <h2>3. Outlier Handling</h2>
    """
    if outlier_info:
        html_content += "<p>Outliers (Z-score > 3) were removed for the following columns:</p><ul>"
        for col, info in outlier_info.items():
            html_content += f"<li>{col}: {info}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No significant outliers (Z-score > 3) were detected and removed in selected columns.</p>"

    html_content += f"""
        </div>

        <div class="section">
            <h2>4. Rule-Based Validation</h2>
    """
    if apply_rules:
        html_content += "<p>Rule-based validation was enabled.</p>"
        if 'Age' in cleaned_df.columns and 'Satisfaction' in cleaned_df.columns:
            html_content += "<p>Specific Rule Applied: 'If Age &lt; 18, Satisfaction was set to blank (NaN).'</p>"
            if rule_applied:
                html_content += "<p>This rule resulted in changes to the 'Satisfaction' column.</p>"
            else:
                html_content += "<p>This rule was applied, but no new blanks were introduced based on the condition.</p>"
        else:
            html_content += "<p>Rule for 'Age' and 'Satisfaction' could not be fully applied (columns missing or renamed).</p>"
    else:
        html_content += "<p>Rule-based validation was not enabled.</p>"

    html_content += f"""
        </div>

        <div class="section">
            <h2>5. Key Statistics for Cleaned Numeric Data</h2>
            {cleaned_df[selected_cols].describe().to_html()}
        </div>

    </body>
    </html>
    """
    
    # Convert HTML to PDF
    pdf_bytes = HTML(string=html_content).write_pdf()
    return pdf_bytes


# --- Main Streamlit App Logic ---
uploaded_file = st.file_uploader("Upload your survey data (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    initial_shape = df.shape # Store initial shape for reporting
    df.columns = df.columns.str.strip()  # Clean column names
    st.success("File uploaded successfully!")
    st.subheader("Preview of Data:")
    st.dataframe(df.head())

    # Step 2: Column Selection for Cleaning
    st.sidebar.header("Data Cleaning Options")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_cols = st.sidebar.multiselect("Select numeric columns for cleaning:", numeric_cols)

    # Missing value imputation
    impute_method = st.sidebar.selectbox("Missing Value Imputation Method:", ["Mean", "Median", "KNN"])

    # Rule-based validation
    st.sidebar.header("Rule-Based Checks")
    apply_rules = st.sidebar.checkbox("Enable basic rule validation (e.g., Age < 18 → Satisfaction must be blank)")

    if st.sidebar.button("Apply Cleaning"):
        cleaned_df = df.copy()
        
        # Track initial missing values for reporting
        initial_missing_report = df[selected_cols].isnull().sum()
        initial_missing_report = initial_missing_report[initial_missing_report > 0]


        if impute_method == "Mean":
            for col in selected_cols:
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
        elif impute_method == "Median":
            for col in selected_cols:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        elif impute_method == "KNN":
            imputer = KNNImputer(n_neighbors=3)
            if selected_cols: # Only apply KNN if there are selected numeric columns
                # KNN imputer requires a DataFrame, so we select only the columns
                # and then assign back to the original DataFrame
                # Use .loc to avoid SettingWithCopyWarning if cleaned_df is a slice
                cleaned_df.loc[:, selected_cols] = pd.DataFrame(imputer.fit_transform(cleaned_df[selected_cols]), 
                                                          columns=selected_cols, 
                                                          index=cleaned_df.index)

        # Outlier handling with Z-score
        outlier_info = {}
        for col in selected_cols:
            if not cleaned_df[col].isnull().all(): # Only apply Z-score if column is not all NaN
                original_count_before_outlier_removal = len(cleaned_df)
                
                # Calculate Z-scores, handling NaNs
                # Create a boolean mask for non-NaN values in the current column
                not_nan_mask = cleaned_df[col].notna()
                
                if not_nan_mask.any(): # Only proceed if there are non-NaN values to compute z-scores on
                    z_scores = np.abs(stats.zscore(cleaned_df.loc[not_nan_mask, col]))
                    
                    # Find the indices of rows that are outliers (where z-score >= 3)
                    # and are also not NaN
                    outlier_row_indices = cleaned_df.loc[not_nan_mask, col][z_scores >= 3].index
                    
                    # Store information about removed outliers before dropping them
                    removed_count_for_col = len(outlier_row_indices)
                    if removed_count_for_col > 0:
                        outlier_info[col] = f"{removed_count_for_col} outliers removed (Z-score >= 3)"
                    
                    # Drop the outlier rows from the DataFrame
                    cleaned_df = cleaned_df.drop(outlier_row_indices)


        # Rule-based check: if Age < 18, Satisfaction must be blank
        rule_applied = False
        if apply_rules and 'Age' in cleaned_df.columns and 'Satisfaction' in cleaned_df.columns:
            original_satisfaction_nans = cleaned_df['Satisfaction'].isnull().sum()
            cleaned_df.loc[(cleaned_df['Age'] < 18), 'Satisfaction'] = np.nan
            if cleaned_df['Satisfaction'].isnull().sum() > original_satisfaction_nans:
                rule_applied = True


        st.subheader("Cleaned Data Preview:")
        st.dataframe(cleaned_df.head())

        # Summary statistics with margin of error
        st.subheader("Unweighted Summary Statistics with Margin of Error:")
        for col in selected_cols:
            if not cleaned_df[col].isnull().all(): # Avoid calculating for all NaN columns
                mean_val = cleaned_df[col].mean()
                # Ensure there's enough data for std_err calculation (at least 2 data points for std dev)
                if len(cleaned_df[col].dropna()) > 1:
                    std_err = cleaned_df[col].std() / np.sqrt(len(cleaned_df[col].dropna()))
                    st.write(f"**{col}** → Mean: {mean_val:.2f}, ± {std_err:.2f} (Margin of Error)")
                else:
                    st.write(f"**{col}** → Mean: {mean_val:.2f}, (Not enough data for Margin of Error)")
            else:
                st.write(f"**{col}** → Not enough data to calculate statistics after cleaning.")


        # --- Report Generation (Text and PDF) ---
        st.subheader("📊 Data Cleaning Report")
        
        # --- Generate Text Report (Existing) ---
        report_content = []
        report_content.append("--- Cleaning Process Summary ---\n")
        report_content.append(f"Date of Report: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append(f"Original Data Dimensions: {initial_shape[0]} rows, {initial_shape[1]} columns\n")
        report_content.append(f"Cleaned Data Dimensions: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns\n")
        report_content.append(f"Selected Numeric Columns for Cleaning: {', '.join(selected_cols) if selected_cols else 'None'}\n")

        report_content.append("\n--- Missing Value Imputation ---\n")
        report_content.append(f"Method Used: {impute_method}\n")
        if not initial_missing_report.empty:
            report_content.append("Initial Missing Values (per selected column):\n")
            for col, count in initial_missing_report.items():
                report_content.append(f"  - {col}: {int(count)} values\n")
        else:
            report_content.append("No initial missing values detected in selected columns.\n")
        
        missing_after_cleaning = cleaned_df[selected_cols].isnull().sum()
        missing_after_cleaning = missing_after_cleaning[missing_after_cleaning > 0]
        if not missing_after_cleaning.empty:
            report_content.append("Missing Values After Imputation/Cleaning (per selected column):\n")
            for col, count in missing_after_cleaning.items():
                report_content.append(f"  - {col}: {int(count)} values (may be due to rule-based checks or outlier removal)\n")
        else:
            report_content.append("No missing values in selected columns after imputation and cleaning.\n")

        report_content.append("\n--- Outlier Handling ---\n")
        if outlier_info:
            report_content.append("Outliers (Z-score > 3) were removed for the following columns:\n")
            for col, info in outlier_info.items():
                report_content.append(f"  - {col}: {info}\n")
        else:
            report_content.append("No significant outliers (Z-score > 3) were detected and removed in selected columns.\n")

        report_content.append("\n--- Rule-Based Validation ---\n")
        if apply_rules:
            report_content.append("Rule-based validation was enabled.\n")
            if 'Age' in df.columns and 'Satisfaction' in df.columns:
                report_content.append("Specific Rule Applied: 'If Age < 18, Satisfaction was set to blank (NaN).'\n")
                if rule_applied:
                    report_content.append("This rule resulted in changes to the 'Satisfaction' column.\n")
                else:
                    report_content.append("This rule was applied, but no new blanks were introduced based on the condition.\n")
            else:
                report_content.append("Rule for 'Age' and 'Satisfaction' could not be fully applied (columns missing).\n")
        else:
            report_content.append("Rule-based validation was not enabled.\n")

        report_content.append("\n--- Key Statistics for Cleaned Numeric Data ---\n")
        report_content.append(cleaned_df[selected_cols].describe().to_string())

        final_report_text = "\n".join(report_content)
        st.text_area("Generated Data Cleaning Report (Text)", value=final_report_text, height=500)

        # Download button for the text report
        st.download_button(
            label="Download Text Report",
            data=final_report_text.encode("utf-8"),
            file_name="data_cleaning_report.txt",
            mime="text/plain"
        )
        
        # --- Generate and Download PDF Report ---
        if WEASYPRINT_AVAILABLE:
            st.write("---")
            st.subheader("Download PDF Report")
            try:
                pdf_bytes = generate_pdf_report(cleaned_df, selected_cols, impute_method, apply_rules, initial_missing_report, outlier_info, rule_applied)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="data_cleaning_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}. Please ensure WeasyPrint and its system dependencies are correctly installed.")
                st.info("On Debian/Ubuntu, you might need: `sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libfontconfig1 libffi-dev libcairo2 libxml2-dev libxslt1-dev`")
        else:
            st.info("Install 'weasyprint' to enable PDF report generation.")

        # --- End Report Generation ---


        def get_table_download_link(df):
            towrite = io.BytesIO()
            df.to_excel(towrite, index=False, header=True)
            towrite.seek(0)
            b64 = base64.b64encode(towrite.read()).decode()
            return f'<a href="data:application/octet-stream;base64,{b64}" download="cleaned_survey.xlsx">📥 Download Cleaned Data</a>'

        st.markdown(get_table_download_link(cleaned_df), unsafe_allow_html=True)

        # Visualization
        st.subheader("Visualizations")
        for col in selected_cols:
            if not cleaned_df[col].isnull().all() and len(cleaned_df[col].dropna()) > 1: # Only plot if there's data and more than 1 non-NaN
                fig, ax = plt.subplots()
                sns.histplot(cleaned_df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
            else:
                st.write(f"No data to plot for {col} after cleaning (or not enough data points).")

    else:
        st.info("Select cleaning options and click 'Apply Cleaning' to process your data.")