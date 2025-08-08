import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import io
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="InsightX - Interactive Smart EDA", layout="wide")

st.title("📊 InsightX - Interactive Smart Exploratory Data Analysis Dashboard")

# Initialize session state for original, cleaned data, and cached reports
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "profile_report_html" not in st.session_state:
    st.session_state.profile_report_html = None

# Helper functions
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]

def calc_skewness(df):
    return df.skew(numeric_only=True)

def cardinality_check(df):
    return df.nunique()

def correlation_issues(df, threshold=0.7):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().abs()
    issues = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i, j] > threshold:
                issues.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    return issues

# Use tabs for better UX
tab_upload, tab_preprocess, tab_visualize, tab_insights = st.tabs(
    ["Upload Data 🗂️", "Preprocess 🧹", "Visualize 📊", "Insights & Report 📈"]
)

with tab_upload:
    st.header("1️⃣ Upload your CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.df_clean = df.copy()
        st.session_state.profile_report_html = None
        st.success("✅ Data uploaded successfully!")
        with st.expander("Show Data Preview & Summary"):
            st.dataframe(df.head())
            st.markdown(f"**Data Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

            # Detect and show column data types
            dtypes = df.dtypes.astype(str)
            col_types = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_types.append("Numeric")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_types.append("Datetime")
                elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
                    col_types.append("Categorical")
                else:
                    col_types.append("Other")
            type_df = pd.DataFrame({"Column": df.columns, "Data Type": dtypes.values, "Detected Type": col_types})
            st.dataframe(type_df)

            st.subheader("Summary Statistics")
            st.dataframe(df.describe(include='all').transpose())

with tab_preprocess:
    st.header("2️⃣ Data Preprocessing")
    if st.session_state.df is None:
        st.warning("Please upload data first in the 'Upload Data' tab.")
    else:
        df = st.session_state.df_clean.copy()

        st.subheader("Missing and Empty Values Summary")
        missing = df.isnull().sum()
        empty_str = (df == '').sum()
        missing_df = pd.DataFrame({"Missing Values": missing, "Empty String Values": empty_str})
        missing_df = missing_df[missing_df.sum(axis=1) > 0]  # only rows with missing or empty
        if missing_df.empty:
            st.info("No missing or empty string values detected.")
        else:
            st.dataframe(missing_df)

            st.subheader("Fill Missing Values")
            fill_option = st.selectbox(
                "Choose method to handle missing values:",
                ["None",
                 "Drop rows with missing values",
                 "Fill with Mean (numeric columns only)",
                 "Fill with Median (numeric columns only)",
                 "Fill with Mode (all columns)",
                 "Fill with Custom Values"]
            )

            custom_values = {}
            if fill_option == "Fill with Custom Values":
                for col in missing_df.index:
                    val = st.text_input(f"Custom fill value for column '{col}': (leave blank to skip)", key=f"custom_{col}")
                    custom_values[col] = val

            if st.button("Apply missing value handling", key="apply_missing"):
                if fill_option == "None":
                    st.info("No changes made to missing values.")
                elif fill_option == "Drop rows with missing values":
                    before_rows = df.shape[0]
                    df.dropna(inplace=True)
                    after_rows = df.shape[0]
                    st.success(f"Dropped {before_rows - after_rows} rows containing missing values.")
                elif fill_option == "Fill with Mean (numeric columns only)":
                    for col in df.select_dtypes(include=np.number).columns:
                        if col in missing_df.index:
                            df[col].fillna(df[col].mean(), inplace=True)
                    st.success("Filled missing numeric values with mean.")
                elif fill_option == "Fill with Median (numeric columns only)":
                    for col in df.select_dtypes(include=np.number).columns:
                        if col in missing_df.index:
                            df[col].fillna(df[col].median(), inplace=True)
                    st.success("Filled missing numeric values with median.")
                elif fill_option == "Fill with Mode (all columns)":
                    for col in missing_df.index:
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col].fillna(mode_val[0], inplace=True)
                    st.success("Filled missing values with mode.")
                elif fill_option == "Fill with Custom Values":
                    for col, val in custom_values.items():
                        if val != "":
                            try:
                                dtype = df[col].dtype
                                if pd.api.types.is_numeric_dtype(dtype):
                                    val_conv = float(val)
                                else:
                                    val_conv = val
                                df[col].fillna(val_conv, inplace=True)
                            except Exception:
                                st.warning(f"Invalid input for column {col}. Skipping.")
                    st.success("Applied custom missing value fills.")

                st.session_state.df_clean = df

        st.subheader("Preview of Preprocessed Data")
        st.dataframe(df.head())

        cleaned_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv")

with tab_visualize:
    st.header("3️⃣ Data Visualization")
    if st.session_state.df_clean is None:
        st.warning("Please upload and preprocess data first.")
    else:
        df = st.session_state.df_clean.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        st.subheader("Auto Visualizations")

        # Layout two columns for side-by-side visuals
        col1, col2 = st.columns(2)

        with col1:
            if numeric_cols:
                st.markdown("### Numeric Data Distributions")
                for col_name in numeric_cols:
                    fig_hist = px.histogram(df, x=col_name, nbins=30, title=f"Histogram of {col_name}", marginal="rug", hover_data=df.columns)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    fig_box = px.box(df, y=col_name, points="outliers", title=f"Boxplot of {col_name}")
                    st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            if categorical_cols:
                st.markdown("### Categorical Data Counts")
                for col_name in categorical_cols:
                    counts = df[col_name].value_counts().reset_index()
                    counts.columns = [col_name, 'Count']
                    fig_bar = px.bar(counts, x=col_name, y='Count', title=f"Bar chart of {col_name}")
                    st.plotly_chart(fig_bar, use_container_width=True)

        # Additional automatic multivariate plots for numeric data
        if len(numeric_cols) > 1:
            st.subheader("Multivariate Numeric Visualizations")
            fig_scatter = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix - Numeric Columns")
            st.plotly_chart(fig_scatter, use_container_width=True)
            corr = df[numeric_cols].corr()
            fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Custom Visualization Area
        st.subheader("Custom Visualization")
        sel_col = st.selectbox("Select column to visualize", df.columns)

        # Detect type
        if sel_col in numeric_cols:
            sel_chart = st.selectbox("Select chart type", ["Histogram", "Boxplot", "Line Chart", "Scatter Plot (select another column)"])
        elif sel_col in categorical_cols:
            sel_chart = st.selectbox("Select chart type", ["Bar Chart", "Pie Chart"])
        else:
            sel_chart = None
            st.info("Selected column not supported for custom visualization.")

        if sel_chart and sel_col:
            if sel_chart == "Histogram":
                fig = px.histogram(df, x=sel_col, nbins=30, title=f"Histogram of {sel_col}")
                st.plotly_chart(fig, use_container_width=True)
            elif sel_chart == "Boxplot":
                fig = px.box(df, y=sel_col, title=f"Boxplot of {sel_col}")
                st.plotly_chart(fig, use_container_width=True)
            elif sel_chart == "Line Chart":
                fig = px.line(df, y=sel_col, title=f"Line Chart of {sel_col}")
                st.plotly_chart(fig, use_container_width=True)
            elif sel_chart == "Scatter Plot (select another column)":
                other_numeric_cols = [c for c in numeric_cols if c != sel_col]
                if other_numeric_cols:
                    x_col = st.selectbox("Select column for X-axis", other_numeric_cols)
                    fig = px.scatter(df, x=x_col, y=sel_col, title=f"Scatter Plot: {sel_col} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least two numeric columns for scatter plot.")
            elif sel_chart == "Bar Chart":
                counts = df[sel_col].value_counts().reset_index()
                counts.columns = [sel_col, "Count"]
                fig = px.bar(counts, x=sel_col, y="Count", title=f"Bar Chart of {sel_col}")
                st.plotly_chart(fig, use_container_width=True)
            elif sel_chart == "Pie Chart":
                fig = px.pie(df, names=sel_col, title=f"Pie Chart of {sel_col}")
                st.plotly_chart(fig, use_container_width=True)

with tab_insights:
    st.header("4️⃣ Data Insights & Report")
    if st.session_state.df_clean is None:
        st.warning("Please upload and preprocess data first.")
    else:
        df = st.session_state.df_clean.copy()

        st.subheader("Null Value Overview")
        null_sum = df.isnull().sum()
        if null_sum.sum() > 0:
            st.dataframe(null_sum[null_sum > 0].sort_values(ascending=False))
        else:
            st.info("No missing values detected.")

        st.subheader("Outliers Detection using IQR")
        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_counts = {}
        for col in numeric_cols:
            outlier_idx = detect_outliers_iqr(df[col].dropna())
            outlier_counts[col] = len(outlier_idx)
        outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=["Outlier Count"])
        st.dataframe(outlier_df)

        st.subheader("Skewness of Numeric Features")
        skewness = calc_skewness(df)
        st.dataframe(skewness)

        st.subheader("Cardinality Check")
        cardinalities = cardinality_check(df)
        st.dataframe(cardinalities)

        st.subheader("Strong Correlation (>|0.7|) Pairs")
        corr_issues = correlation_issues(df)
        if corr_issues:
            corr_df = pd.DataFrame(corr_issues, columns=["Feature 1", "Feature 2", "Correlation"])
            st.dataframe(corr_df)
        else:
            st.info("No strong correlations detected.")

        # Generate insights summary text for download
        insights_str = io.StringIO()
        insights_str.write("InsightX Data Insights Summary\n")
        insights_str.write("===============================\n\n")
        insights_str.write("Null Values:\n")
        for col, val in null_sum[null_sum > 0].items():
            insights_str.write(f"{col}: {val}\n")
        insights_str.write("\nOutlier Counts (IQR) per Numeric Column:\n")
        for col, val in outlier_counts.items():
            insights_str.write(f"{col}: {val}\n")
        insights_str.write("\nSkewness of Numeric Columns:\n")
        for col, val in skewness.dropna().items():
            insights_str.write(f"{col}: {val:.4f}\n")
        insights_str.write("\nCardinality per Column:\n")
        for col, val in cardinalities.items():
            insights_str.write(f"{col}: {val}\n")
        if corr_issues:
            insights_str.write("\nStrongly Correlated Pairs (>|0.7|):\n")
            for c1, c2, corr_val in corr_issues:
                insights_str.write(f"{c1} & {c2}: {corr_val:.4f}\n")
        else:
            insights_str.write("\nNo strong correlations detected.\n")

        # Display profiling report with caching
        st.subheader("Profiling Report")
        if st.session_state.profile_report_html is None:
            with st.spinner("Generating profiling report..."):
                profile = ProfileReport(df, title="InsightX Profiler", explorative=True)
                st.session_state.profile_report_html = profile.to_html()
                st_profile_report(profile)
        else:
            st.markdown("Profiling report loaded from cache.")
            st_profile_report(ProfileReport(df, title="InsightX Profiler", explorative=True))

        # Download cleaned CSV
        cleaned_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv_report")

        # Download insights summary text
        st.download_button("📥 Download Insights Summary (TXT)", insights_str.getvalue(), "insights_summary.txt", "text/plain", key="download_insights_summary")

        # Download example visualization: Correlation Heatmap as PNG
        st.subheader("Download Example Visualization: Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8, 6))
        # sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)

        plt.title("Correlation Heatmap")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("📥 Download Correlation Heatmap", buf, "correlation_heatmap.png", "image/png", key="download_corr_heatmap")
        plt.close(fig)
