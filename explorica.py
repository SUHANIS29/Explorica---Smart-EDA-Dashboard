import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# -------------------------------
# Page config and title
# -------------------------------
st.set_page_config(page_title="Explorica - Smart EDA", layout="wide")
st.title("📊 Explorica - Smart Exploratory Data Analysis Dashboard")

# -------------------------------
# Session state initialization
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "profile_report" not in st.session_state:
    st.session_state.profile_report = None
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = []
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None
if "pending_step" not in st.session_state:
    st.session_state.pending_step = None

# -------------------------------
# Utility functions
# -------------------------------
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
    if numeric_df.shape[1] < 2:
        return []
    corr = numeric_df.corr()
    corr_abs = corr.abs()
    issues = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr_abs.iloc[i, j] > threshold:
                issues.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    return issues

def normalize_missing(dfx):
    dfx = dfx.copy()
    for c in dfx.select_dtypes(include=["object"]).columns:
        dfx[c] = dfx[c].astype(str).str.strip()
        dfx[c] = dfx[c].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return dfx

def current_view_df():
    # Always use preview if available; otherwise use the cleaned dataset
    return st.session_state.df_preview if st.session_state.df_preview is not None else st.session_state.df_clean

# -------------------------------
# Tabs
# -------------------------------
tab_upload, tab_preprocess, tab_visualize, tab_insights = st.tabs(
    ["Upload Data 🗂️", "Preprocess 🧹", "Visualize 📊", "Insights & Report 📈"]
)

# -------------------------------
# Upload Tab
# -------------------------------
with tab_upload:
    st.header("1️⃣ Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = normalize_missing(df)

        st.session_state.df = df.copy()
        st.session_state.df_clean = df.copy()
        st.session_state.profile_report = None
        st.session_state.pipeline_steps = []
        st.session_state.df_preview = None
        st.session_state.pending_step = None

        st.success("File uploaded successfully!")
        with st.expander("Data Preview and Summary", expanded=True):
            st.dataframe(df.head())
            st.write(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")

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

            dtype_df = pd.DataFrame({"Column": df.columns, "Dtype": dtypes.values, "Detected Type": col_types})
            st.dataframe(dtype_df)
            st.write(df.describe(include="all").transpose())

# -------------------------------
# Preprocess Tab
# -------------------------------
with tab_preprocess:
    st.header("2️⃣ Preprocessing")
    if st.session_state.df is None:
        st.warning("Upload dataset first.")
    else:
        df_temp = st.session_state.df_clean.copy()

        # st.markdown("### A) Data Hygiene")
        # with st.container(border=True):
        #     st.write("Duplicates")
        #     dup_subset_cols = st.multiselect("Columns to check duplicates on", options=df_temp.columns, key="dup_subset_cols")
        #     duplicates = df_temp.duplicated(subset=dup_subset_cols if dup_subset_cols else None).sum()
        #     st.write(f"Duplicate rows: {duplicates}")

        #     if duplicates > 0:
        #         if st.button("Remove Duplicates (Preview)", key="remove_dup_preview"):
        #             df_preview = df_temp.drop_duplicates(subset=dup_subset_cols if dup_subset_cols else None)
        #             st.session_state.df_preview = df_preview
        #             st.session_state.pending_step = {
        #                 "step": "drop_duplicates",
        #                 "subset": dup_subset_cols,
        #                 "message": f"Removed {df_temp.shape[0] - df_preview.shape[0]} duplicate rows."
        #             }
        #             st.info("Preview created. Click 'Save Changes' to finalize.")

        #         if st.session_state.df_preview is not None and st.session_state.pending_step and st.session_state.pending_step.get("step") == "drop_duplicates":
        #             if st.button("💾 Save Duplicate Removal Changes", key="save_dup_changes"):
        #                 st.session_state.df_clean = st.session_state.df_preview.copy()
        #                 st.session_state.pipeline_steps.append(st.session_state.pending_step)
        #                 st.session_state.df_preview = None
        #                 st.session_state.pending_step = None
        #                 st.session_state.profile_report = None  # invalidate profile cache
        #                 st.success("Duplicate removal saved.")
        #                 st.rerun()

        #     st.write("Column Operations")
        #     drop_cols = st.multiselect("Columns to drop", options=df_temp.columns, key="drop_cols")
        #     if drop_cols and st.button("Drop selected columns (Preview)", key="drop_selected_cols_preview"):
        #         df_preview = df_temp.drop(columns=drop_cols)
        #         st.session_state.df_preview = df_preview
        #         st.session_state.pending_step = {"step": "drop_columns", "columns": drop_cols, "message": f"Dropped columns: {drop_cols}"}
        #         st.info("Preview created. Click 'Save Changes' to finalize.")

        #     if st.session_state.df_preview is not None and st.session_state.pending_step and st.session_state.pending_step.get("step") == "drop_columns":
        #         if st.button("💾 Save Column Drop Changes", key="save_drop_changes"):
        #             st.session_state.df_clean = st.session_state.df_preview.copy()
        #             st.session_state.pipeline_steps.append(st.session_state.pending_step)
        #             st.session_state.df_preview = None
        #             st.session_state.pending_step = None
        #             st.session_state.profile_report = None  # invalidate profile cache
        #             st.success("Column drop saved.")
        #             st.rerun()

        st.markdown("### Data Preprocessing Steps")
        with st.container(border=True):
            df_base = current_view_df().copy()
            missing_counts = df_base.isna().sum()
            missing_display = missing_counts[missing_counts > 0].to_frame("Missing")

            if missing_display.empty:
                st.info("No missing values detected.")
            else:
                st.dataframe(missing_display)
                imputation_choices = {}
                for col in missing_display.index:
                    st.markdown(f"**Column: `{col}`**")
                    col_type = pd.api.types.infer_dtype(df_base[col])

                    if col_type in ['integer', 'floating']:
                        method = st.selectbox(
                            f"Select method for {col}",
                            ["Drop Rows with any NaN", "Mean", "Median", "Constant", "Do Nothing"],
                            key=f"impute_num_{col}"
                        )
                        constant_val = None
                        if method == "Constant":
                            constant_val = st.number_input(f"Constant value for {col}", key=f"constant_num_{col}", value=0.0)
                        imputation_choices[col] = {"method": method, "value": constant_val}

                    elif col_type in ['string', 'object', 'category']:
                        method = st.selectbox(
                            f"Select method for {col}",
                            ["Drop Rows with any NaN", "Mode", "Constant", "Do Nothing"],
                            key=f"impute_cat_{col}"
                        )
                        constant_val = None
                        if method == "Constant":
                            constant_val = st.text_input(f"Constant value for {col}", key=f"constant_cat_{col}", value="missing")
                        imputation_choices[col] = {"method": method, "value": constant_val}

                    else:
                        st.info(f"Skipping `{col}` as it is of type `{col_type}` and not supported for imputation.")

                if st.button("Apply changes", key="apply_missing_treatment_preview"):
                    df_to_impute = current_view_df().copy()
                    msg_parts = []

                    for col, choices in imputation_choices.items():
                        method = choices["method"]
                        value = choices["value"]

                        if method == "Drop Rows with any NaN":
                            df_to_impute.dropna(subset=[col], inplace=True)
                            msg_parts.append(f"Dropped rows with missing values in '{col}'.")
                        elif method == "Mean":
                            df_to_impute[col].fillna(df_to_impute[col].mean(), inplace=True)
                            msg_parts.append(f"Imputed '{col}' with Mean.")
                        elif method == "Median":
                            df_to_impute[col].fillna(df_to_impute[col].median(), inplace=True)
                            msg_parts.append(f"Imputed '{col}' with Median.")
                        elif method == "Mode":
                            mode_val = df_to_impute[col].mode()
                            if not mode_val.empty:
                                df_to_impute[col].fillna(mode_val[0], inplace=True)
                                msg_parts.append(f"Imputed '{col}' with Mode.")
                        elif method == "Constant" and value is not None:
                            df_to_impute[col].fillna(value, inplace=True)
                            msg_parts.append(f"Imputed '{col}' with constant '{value}'.")

                    msg = " | ".join(msg_parts) if msg_parts else "No changes applied."
                    st.session_state.df_preview = df_to_impute
                    st.session_state.pending_step = {
                        "step": "impute",
                        "params": {"imputation_choices": imputation_choices},
                        "message": msg,
                    }
                    st.info(f"Preview created. Changes: {msg}")

        # Global preview controls
        if st.session_state.df_preview is not None:
            st.markdown("---")
            st.markdown("#### Preview (Not yet saved)")
            st.dataframe(st.session_state.df_preview.head(), use_container_width=True)
            st.write(f"Preview Shape: {st.session_state.df_preview.shape[0]} rows, {st.session_state.df_preview.shape[1]} columns")
            if st.session_state.pending_step:
                st.write(f"Pending Action: {st.session_state.pending_step['message']}")

            col_save, col_discard = st.columns(2)
            with col_save:
                if st.button("💾 Save Changes", key="save_preview_result"):
                    st.session_state.df_clean = st.session_state.df_preview.copy()
                    st.session_state.pipeline_steps.append(st.session_state.pending_step)
                    st.session_state.df_preview = None
                    st.session_state.pending_step = None
                    st.session_state.profile_report = None  # invalidate profile cache
                    st.success("Changes saved successfully!")
                    st.rerun()

            with col_discard:
                if st.button("↩️ Discard Preview", key="discard_preview"):
                    st.session_state.df_preview = None
                    st.session_state.pending_step = None
                    st.info("Preview discarded. No changes saved.")
                    st.rerun()

        st.markdown("---")
        st.markdown("### Download")
        final_df = st.session_state.df_clean.copy()
        st.write(f"Current Data Shape: **{final_df.shape[0]} rows** | **{final_df.shape[1]} columns**")
        # st.dataframe(final_df.head(), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Reset to original upload", key="reset_to_original"):
                st.session_state.df_clean = st.session_state.df.copy()
                st.session_state.profile_report = None
                st.session_state.pipeline_steps = []
                st.session_state.df_preview = None
                st.session_state.pending_step = None
                st.success("Reset to original dataset.")
                st.rerun()

        with c2:
            if st.button("Recompute profiling report", key="recompute_profile"):
                st.session_state.profile_report = None
                st.info("Profiling report will regenerate in the Insights tab.")
                st.rerun()

        with c3:
            if st.button("Show pipeline steps", key="show_pipeline"):
                if st.session_state.pipeline_steps:
                    st.json(st.session_state.pipeline_steps)
                else:
                    st.info("No recorded steps yet.")

        cleaned_csv = final_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download Current View", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_preproc")

# -------------------------------
# Visualize Tab
# -------------------------------
with tab_visualize:
    st.header("3️⃣ Visualization")
    if st.session_state.df_clean is None:
        st.warning("Please upload and preprocess data first.")
    else:
        # Use current working view (preview if active, else cleaned)
        df = current_view_df().copy()

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Numeric columns visualization")
            for i, col in enumerate(num_cols):
                hist_fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
                st.plotly_chart(hist_fig, use_container_width=True, key=f"hist_{col}_{i}")

                box_fig = px.box(df, y=col, points='outliers', title=f"Box plot of {col}")
                st.plotly_chart(box_fig, use_container_width=True, key=f"box_{col}_{i}")

        with col2:
            st.subheader("Categorical columns visualization")
            for i, col in enumerate(cat_cols):
                count_df = df[col].value_counts(dropna=False).reset_index()
                count_df.columns = [col, 'count']
                bar_fig = px.bar(count_df, x=col, y='count', title=f"Bar chart of {col}")
                st.plotly_chart(bar_fig, use_container_width=True, key=f"bar_{col}_{i}")

        if len(num_cols) > 1:
            st.subheader("Pairwise numeric visualization")
            pair_fig = px.scatter_matrix(df, dimensions=num_cols)
            st.plotly_chart(pair_fig, use_container_width=True, key="pairwise_numeric")

            corr_matrix = df[num_cols].corr()
            corr_fig = px.imshow(corr_matrix, text_auto=True, title="Correlation heatmap")
            st.plotly_chart(corr_fig, use_container_width=True, key="correlation_heatmap")

# -------------------------------
# Insights & Report Tab
# -------------------------------
with tab_insights:
    st.header("4️⃣ Insights and Final Report")
    if st.session_state.df_clean is None:
        st.warning("Please upload and preprocess data first.")
    else:
        # Always use the current working view
        df = current_view_df().copy()
        numeric_df = df.select_dtypes(include=np.number)

        st.subheader("Missing Values Overview")
        missing_vals = df.isnull().sum()
        missing_vals = missing_vals[missing_vals > 0]
        if not missing_vals.empty:
            st.dataframe(missing_vals)
        else:
            st.info("No missing values detected.")

        st.subheader("Outlier Detection (IQR method)")
        outlier_counts = {}
        for col in numeric_df.columns:
            outlier_indices = detect_outliers_iqr(numeric_df[col].dropna())
            outlier_counts[col] = len(outlier_indices)
        outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outlier Count'])
        st.dataframe(outlier_df)

        st.subheader("Skewness")
        skewness = calc_skewness(df)
        st.dataframe(skewness)

        st.subheader("Cardinality")
        cardinality = cardinality_check(df)
        st.dataframe(cardinality)

        st.subheader("Highly Correlated Pairs (|corr| > 0.7)")
        corr_pairs = correlation_issues(df)
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
            st.dataframe(corr_df)
        else:
            st.info("No highly correlated pairs found.")

        st.subheader("Profiling Report")
        df_for_profile = df.copy()
        in_preview = st.session_state.df_preview is not None

        if in_preview:
            # Always regenerate for preview to reflect unsaved changes
            with st.spinner("Generating profiling report (preview)..."):
                profile = ProfileReport(df_for_profile, title="Explorica Profiling Report (Preview)", explorative=True)
            st_profile_report(profile)
        else:
            # Use cached report for df_clean; regenerate if missing or invalidated
            if st.session_state.profile_report is None:
                with st.spinner("Generating profiling report..."):
                    profile = ProfileReport(df_for_profile, title="Explorica Profiling Report", explorative=True)
                st.session_state.profile_report = profile
            st_profile_report(st.session_state.profile_report)
