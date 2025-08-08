# # # # # # # # import streamlit as st
# # # # # # # # import pandas as pd
# # # # # # # # import plotly.express as px
# # # # # # # # from ydata_profiling import ProfileReport
# # # # # # # # from streamlit_pandas_profiling import st_profile_report

# # # # # # # # st.set_page_config(page_title="Explorica - Smart EDA", layout="wide")
# # # # # # # # st.sidebar.title("📊 Explorica EDA Dashboard")

# # # # # # # # # Sidebar navigation
# # # # # # # # section = st.sidebar.radio("Go to", ["Upload Data", "Preprocess", "Visualize", "Insights & Report"])

# # # # # # # # # Initialize session state
# # # # # # # # if "df" not in st.session_state:
# # # # # # # #     st.session_state.df = None

# # # # # # # # # Upload Data
# # # # # # # # if section == "Upload Data":
# # # # # # # #     st.header("📤 Upload your CSV file")
# # # # # # # #     file = st.file_uploader("Choose CSV", type=["csv"])
# # # # # # # #     if file:
# # # # # # # #         df = pd.read_csv(file)
# # # # # # # #         st.session_state.df = df
# # # # # # # #         st.success("Data uploaded successfully!")
# # # # # # # #         st.dataframe(df.head())

# # # # # # # # # Preprocess Data
# # # # # # # # elif section == "Preprocess":
# # # # # # # #     st.header("🧹 Preprocess your Data")
# # # # # # # #     if st.session_state.df is not None:
# # # # # # # #         df = st.session_state.df.copy()
# # # # # # # #         st.write("Shape:", df.shape)
# # # # # # # #         st.write("Null values:", df.isnull().sum())
# # # # # # # #         if st.button("Drop Nulls"):
# # # # # # # #             df.dropna(inplace=True)
# # # # # # # #             st.session_state.df = df
# # # # # # # #             st.success("Null values dropped")
# # # # # # # #         st.dataframe(df)
# # # # # # # #         st.download_button("📥 Download Cleaned CSV", df.to_csv(index=False), "cleaned_data.csv", "text/csv")
# # # # # # # #     else:
# # # # # # # #         st.warning("Please upload data first.")

# # # # # # # # # Visualize Data
# # # # # # # # elif section == "Visualize":
# # # # # # # #     st.header("📊 Data Visualization")
# # # # # # # #     if st.session_state.df is not None:
# # # # # # # #         df = st.session_state.df
# # # # # # # #         col = st.selectbox("Select column to visualize", df.columns)
# # # # # # # #         chart_type = st.selectbox("Choose chart type", ["Histogram", "Bar", "Line"])
# # # # # # # #         if chart_type == "Histogram":
# # # # # # # #             fig = px.histogram(df, x=col)
# # # # # # # #         elif chart_type == "Bar":
# # # # # # # #             fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col)
# # # # # # # #         else:
# # # # # # # #             fig = px.line(df, y=col)
# # # # # # # #         st.plotly_chart(fig, use_container_width=True)
# # # # # # # #     else:
# # # # # # # #         st.warning("Please upload data first.")

# # # # # # # # # Insights & Report
# # # # # # # # elif section == "Insights & Report":
# # # # # # # #     st.header("📈 Smart Insights & Report")
# # # # # # # #     if st.session_state.df is not None:
# # # # # # # #         df = st.session_state.df
# # # # # # # #         pr = ProfileReport(df, title="Explorica Report", explorative=True)
# # # # # # # #         st_profile_report(pr)
# # # # # # # #     else:
# # # # # # # #         st.warning("Please upload data first.")

# # # # # # #----visualization tab-------


# # # # # # # import streamlit as st
# # # # # # # import pandas as pd
# # # # # # # import numpy as np
# # # # # # # import plotly.express as px
# # # # # # # import seaborn as sns
# # # # # # # import matplotlib.pyplot as plt
# # # # # # # import io

# # # # # # # st.set_page_config(page_title="📊 Advanced EDA Dashboard", layout="wide")
# # # # # # # st.sidebar.title("📊 EDA Dashboard Navigation")

# # # # # # # section = st.sidebar.radio("Go to", ["Upload Data", "Preprocess Data", "Visualizations", "Insights"])

# # # # # # # if "df" not in st.session_state:
# # # # # # #     st.session_state.df = None


# # # # # # # # ------------------ 1) UPLOAD TAB ------------------
# # # # # # # if section == "Upload Data":
# # # # # # #     st.header("📤 Upload your CSV file")
# # # # # # #     file = st.file_uploader("Choose CSV file", type=["csv"])

# # # # # # #     if file:
# # # # # # #         df = pd.read_csv(file)
# # # # # # #         st.session_state.df = df

# # # # # # #         st.success("✅ Data uploaded successfully!")

# # # # # # #         # Preview
# # # # # # #         st.subheader("🔍 Data Preview")
# # # # # # #         st.dataframe(df.head())

# # # # # # #         # Dimensions
# # # # # # #         st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# # # # # # #         # Column types
# # # # # # #         col_types = df.dtypes.astype(str).to_dict()
# # # # # # #         st.write("**Column Data Types:**", col_types)

# # # # # # #         # Auto column type detection
# # # # # # #         num_cols = df.select_dtypes(include=np.number).columns.tolist()
# # # # # # #         cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
# # # # # # #         st.write("**Detected Numeric Columns:**", num_cols)
# # # # # # #         st.write("**Detected Categorical Columns:**", cat_cols)

# # # # # # #         # Summary statistics
# # # # # # #         st.subheader("📊 Summary Statistics")
# # # # # # #         st.dataframe(df.describe())


# # # # # # # # ------------------ 2) PREPROCESSING TAB ------------------
# # # # # # # elif section == "Preprocess Data":
# # # # # # #     st.header("🧹 Preprocess Data")
# # # # # # #     if st.session_state.df is not None:
# # # # # # #         df = st.session_state.df.copy()

# # # # # # #         # Missing values summary
# # # # # # #         st.subheader("📌 Missing Value Summary")
# # # # # # #         missing = df.isnull().sum()
# # # # # # #         missing = missing[missing > 0]
# # # # # # #         st.write(missing)

# # # # # # #         # Fill missing values
# # # # # # #         st.subheader("🛠 Fill Missing Values")
# # # # # # #         if not missing.empty:
# # # # # # #             col_to_fill = st.selectbox("Select Column to Fill", missing.index)
# # # # # # #             method = st.selectbox("Fill Method", ["Mean", "Median", "Mode", "Custom Value"])

# # # # # # #             if method == "Mean":
# # # # # # #                 df[col_to_fill].fillna(df[col_to_fill].mean(), inplace=True)
# # # # # # #             elif method == "Median":
# # # # # # #                 df[col_to_fill].fillna(df[col_to_fill].median(), inplace=True)
# # # # # # #             elif method == "Mode":
# # # # # # #                 df[col_to_fill].fillna(df[col_to_fill].mode()[0], inplace=True)
# # # # # # #             elif method == "Custom Value":
# # # # # # #                 custom_val = st.text_input("Enter Custom Value")
# # # # # # #                 if custom_val != "":
# # # # # # #                     df[col_to_fill].fillna(custom_val, inplace=True)

# # # # # # #             st.success(f"Missing values in {col_to_fill} filled using {method}")

# # # # # # #         # Preview cleaned data
# # # # # # #         st.subheader("📄 Cleaned Data Preview")
# # # # # # #         st.dataframe(df.head())

# # # # # # #         # Download cleaned CSV
# # # # # # #         csv_buffer = io.StringIO()
# # # # # # #         df.to_csv(csv_buffer, index=False)
# # # # # # #         st.download_button("📥 Download Cleaned Data", csv_buffer.getvalue(), "cleaned_data.csv", "text/csv")

# # # # # # #         st.session_state.df = df
# # # # # # #     else:
# # # # # # #         st.warning("Please upload data first.")


# # # # # # # # ------------------ 3) VISUALIZATION TAB ------------------
# # # # # # # elif section == "Visualizations":
# # # # # # #     st.header("📊 Auto Visualizations")
# # # # # # #     if st.session_state.df is not None:
# # # # # # #         df = st.session_state.df
# # # # # # #         num_cols = df.select_dtypes(include=np.number).columns.tolist()
# # # # # # #         cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# # # # # # #         # Numeric column visualizations
# # # # # # #         st.subheader("📈 Numeric Columns")
# # # # # # #         for col in num_cols:
# # # # # # #             fig = px.histogram(df, x=col, title=f"Histogram of {col}")
# # # # # # #             st.plotly_chart(fig, use_container_width=True)

# # # # # # #             fig_box = px.box(df, y=col, title=f"Boxplot of {col}")
# # # # # # #             st.plotly_chart(fig_box, use_container_width=True)

# # # # # # #         # Categorical column visualizations
# # # # # # #         st.subheader("📊 Categorical Columns")
# # # # # # #         for col in cat_cols:
# # # # # # #             fig_bar = px.bar(df[col].value_counts().reset_index(), x='index', y=col,
# # # # # # #                              title=f"Bar Chart of {col}")
# # # # # # #             st.plotly_chart(fig_bar, use_container_width=True)

# # # # # # #             fig_pie = px.pie(df, names=col, title=f"Pie Chart of {col}")
# # # # # # #             st.plotly_chart(fig_pie, use_container_width=True)

# # # # # # #     else:
# # # # # # #         st.warning("Please upload data first.")


# # # # # # # # ------------------ 4) INSIGHTS TAB ------------------
# # # # # # # elif section == "Insights":
# # # # # # #     st.header("📈 Data Insights")
# # # # # # #     if st.session_state.df is not None:
# # # # # # #         df = st.session_state.df
# # # # # # #         num_cols = df.select_dtypes(include=np.number).columns.tolist()

# # # # # # #         # Missing value check
# # # # # # #         st.subheader("📌 Missing Value Check")
# # # # # # #         st.write(df.isnull().sum())

# # # # # # #         # Outlier detection (IQR method)
# # # # # # #         st.subheader("🚨 Outlier Detection (IQR Method)")
# # # # # # #         outlier_summary = {}
# # # # # # #         for col in num_cols:
# # # # # # #             Q1 = df[col].quantile(0.25)
# # # # # # #             Q3 = df[col].quantile(0.75)
# # # # # # #             IQR = Q3 - Q1
# # # # # # #             outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
# # # # # # #             outlier_summary[col] = int(outliers)
# # # # # # #         st.write(outlier_summary)

# # # # # # #         # Skewness detection
# # # # # # #         st.subheader("📏 Skewness of Numeric Columns")
# # # # # # #         skewness = df[num_cols].skew().to_dict()
# # # # # # #         st.write(skewness)

# # # # # # #         # Correlation matrix
# # # # # # #         st.subheader("🔗 Correlation Insights")
# # # # # # #         corr = df[num_cols].corr()
# # # # # # #         st.dataframe(corr)
# # # # # # #         fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
# # # # # # #         st.plotly_chart(fig_corr, use_container_width=True)

# # # # # # #         # Cardinality check
# # # # # # #         st.subheader("🔢 Cardinality of Categorical Columns")
# # # # # # #         card = {col: df[col].nunique() for col in df.select_dtypes(exclude=np.number).columns}
# # # # # # #         st.write(card)

# # # # # # #         # Download insights report
# # # # # # #         insights_buffer = io.StringIO()
# # # # # # #         insights_buffer.write("Missing Values:\n")
# # # # # # #         insights_buffer.write(str(df.isnull().sum()))
# # # # # # #         insights_buffer.write("\n\nOutliers:\n")
# # # # # # #         insights_buffer.write(str(outlier_summary))
# # # # # # #         insights_buffer.write("\n\nSkewness:\n")
# # # # # # #         insights_buffer.write(str(skewness))
# # # # # # #         insights_buffer.write("\n\nCorrelation:\n")
# # # # # # #         insights_buffer.write(str(corr))
# # # # # # #         insights_buffer.write("\n\nCardinality:\n")
# # # # # # #         insights_buffer.write(str(card))

# # # # # # #         st.download_button("📥 Download Insights Report", insights_buffer.getvalue(), "insights.txt", "text/plain")
# # # # # # #     else:
# # # # # # #         st.warning("Please upload data first.")


# # # # # # import streamlit as st
# # # # # # import pandas as pd
# # # # # # import numpy as np
# # # # # # import plotly.express as px
# # # # # # from ydata_profiling import ProfileReport
# # # # # # from streamlit_pandas_profiling import st_profile_report
# # # # # # import io
# # # # # # import seaborn as sns
# # # # # # import matplotlib.pyplot as plt

# # # # # # st.set_page_config(page_title="Explorica - Smart EDA Dashboard", layout="wide")

# # # # # # st.title("📊 Explorica - Smart Exploratory Data Analysis")

# # # # # # # Initialize session state for original and cleaned data and insights
# # # # # # if "df" not in st.session_state:
# # # # # #     st.session_state.df = None
# # # # # # if "df_clean" not in st.session_state:
# # # # # #     st.session_state.df_clean = None

# # # # # # # Helper functions for insights
# # # # # # def detect_outliers_iqr(data):
# # # # # #     Q1 = np.percentile(data, 25)
# # # # # #     Q3 = np.percentile(data, 75)
# # # # # #     IQR = Q3 - Q1
# # # # # #     lower_bound = Q1 - 1.5 * IQR
# # # # # #     upper_bound = Q3 + 1.5 * IQR
# # # # # #     return np.where((data < lower_bound) | (data > upper_bound))[0]

# # # # # # def calc_skewness_desc(df):
# # # # # #     return df.skew(numeric_only=True)

# # # # # # def cardinality_check(df):
# # # # # #     return df.nunique()

# # # # # # def correlation_issues(df, threshold=0.7):
# # # # # #     corr = df.corr().abs()
# # # # # #     high_corr_pairs = []
# # # # # #     for i in range(len(corr.columns)):
# # # # # #         for j in range(i+1, len(corr.columns)):
# # # # # #             if corr.iloc[i, j] > threshold:
# # # # # #                 col_i = corr.columns[i]
# # # # # #                 col_j = corr.columns[j]
# # # # # #                 high_corr_pairs.append((col_i, col_j, corr.iloc[i, j]))
# # # # # #     return high_corr_pairs

# # # # # # # Sidebar navigation
# # # # # # section = st.sidebar.radio("Navigation", ["Upload Data", "Preprocess", "Visualize", "Insights & Report"])

# # # # # # # --- 1. Upload Data ---
# # # # # # if section == "Upload Data":
# # # # # #     st.header("1️⃣ Upload CSV File")
# # # # # #     uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
# # # # # #     if uploaded_file is not None:
# # # # # #         # Load data
# # # # # #         df = pd.read_csv(uploaded_file)
# # # # # #         st.session_state.df = df
# # # # # #         st.session_state.df_clean = df.copy()
# # # # # #         st.success("Data uploaded successfully!")
        
# # # # # #         # Basic Data Info
# # # # # #         st.subheader("Data Preview")
# # # # # #         st.dataframe(df.head())
        
# # # # # #         st.subheader("Data Dimensions")
# # # # # #         st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
# # # # # #         st.subheader("Column Data Types & Detected Types")
# # # # # #         dtypes = df.dtypes.astype(str)
# # # # # #         # Detect column types a bit more explicitly
# # # # # #         col_type_detected = []
# # # # # #         for col in df.columns:
# # # # # #             if pd.api.types.is_numeric_dtype(df[col]):
# # # # # #                 col_type_detected.append("Numeric")
# # # # # #             elif pd.api.types.is_datetime64_any_dtype(df[col]):
# # # # # #                 col_type_detected.append("Datetime")
# # # # # #             elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
# # # # # #                 col_type_detected.append("Categorical")
# # # # # #             else:
# # # # # #                 col_type_detected.append("Other")
# # # # # #         type_df = pd.DataFrame({"Column": df.columns, "Data Type": dtypes.values, "Detected Type": col_type_detected})
# # # # # #         st.dataframe(type_df)
        
# # # # # #         st.subheader("Summary Statistics")
# # # # # #         st.dataframe(df.describe(include='all').transpose())

# # # # # # # --- 2. Preprocess ---
# # # # # # elif section == "Preprocess":
# # # # # #     st.header("2️⃣ Data Preprocessing")
# # # # # #     if st.session_state.df is None:
# # # # # #         st.warning("Please upload data first in Upload Data tab.")
# # # # # #     else:
# # # # # #         df = st.session_state.df_clean.copy()
        
# # # # # #         # Show missing and null values summary
# # # # # #         missing_count = df.isnull().sum()
# # # # # #         empty_count = (df == '').sum()
        
# # # # # #         st.subheader("Missing and Empty Value Summary")
# # # # # #         missing_df = pd.DataFrame({"Missing Values": missing_count, "Empty String Values": empty_count})
# # # # # #         missing_df = missing_df[missing_df.sum(axis=1) > 0]
        
# # # # # #         if missing_df.empty:
# # # # # #             st.info("No missing or empty string values detected.")
# # # # # #         else:
# # # # # #             st.dataframe(missing_df)
            
# # # # # #             # Missing value filling options
# # # # # #             st.subheader("Fill Missing Values")
# # # # # #             fill_method = st.selectbox("Select method to fill missing values", 
# # # # # #                                        ["None", "Fill with Mean (numeric columns only)", "Fill with Median (numeric only)", "Fill with Mode (all columns)", "Fill with Custom Value"])

# # # # # #             if fill_method == "Fill with Custom Value":
# # # # # #                 custom_fill = {}
# # # # # #                 for col in missing_df.index:
# # # # # #                     val = st.text_input(f"Fill value for column '{col}' (leave blank to skip):", key=f"fill_{col}")
# # # # # #                     custom_fill[col] = val
                
# # # # # #             if st.button("Apply Missing Value Treatment"):
# # # # # #                 if fill_method == "None":
# # # # # #                     st.info("No changes applied to missing values.")
# # # # # #                 elif fill_method == "Fill with Mean (numeric columns only)":
# # # # # #                     for col in df.select_dtypes(include=[np.number]).columns:
# # # # # #                         if col in missing_df.index:
# # # # # #                             df[col].fillna(df[col].mean(), inplace=True)
# # # # # #                     st.success("Filled missing numeric values with mean.")
# # # # # #                 elif fill_method == "Fill with Median (numeric only)":
# # # # # #                     for col in df.select_dtypes(include=[np.number]).columns:
# # # # # #                         if col in missing_df.index:
# # # # # #                             df[col].fillna(df[col].median(), inplace=True)
# # # # # #                     st.success("Filled missing numeric values with median.")
# # # # # #                 elif fill_method == "Fill with Mode (all columns)":
# # # # # #                     for col in missing_df.index:
# # # # # #                         mode_val = df[col].mode()
# # # # # #                         if not mode_val.empty:
# # # # # #                             df[col].fillna(mode_val[0], inplace=True)
# # # # # #                     st.success("Filled missing values with mode.")
# # # # # #                 elif fill_method == "Fill with Custom Value":
# # # # # #                     for col, val in custom_fill.items():
# # # # # #                         if val != "":
# # # # # #                             dtype = df[col].dtype
# # # # # #                             # Try to convert to appropriate dtype
# # # # # #                             try:
# # # # # #                                 if pd.api.types.is_numeric_dtype(dtype):
# # # # # #                                     val_conv = float(val)
# # # # # #                                 else:
# # # # # #                                     val_conv = val
# # # # # #                             except:
# # # # # #                                 val_conv = val
# # # # # #                             df[col].fillna(val_conv, inplace=True)
# # # # # #                     st.success("Filled missing values with custom values.")
# # # # # #                 st.session_state.df_clean = df

# # # # # #         # After filling missing values, show preview
# # # # # #         st.subheader("Data Preview After Preprocessing")
# # # # # #         st.dataframe(df.head())
        
# # # # # #         # Download cleaned data CSV
# # # # # #         cleaned_csv = df.to_csv(index=False).encode('utf-8')
# # # # # #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

# # # # # # # --- 3. Visualization ---
# # # # # # elif section == "Visualize":
# # # # # #     st.header("3️⃣ Data Visualization")
# # # # # #     if st.session_state.df_clean is None:
# # # # # #         st.warning("Please upload and preprocess data first.")
# # # # # #     else:
# # # # # #         df = st.session_state.df_clean
# # # # # #         numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# # # # # #         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# # # # # #         st.subheader("Auto Visualizations")

# # # # # #         # Numeric columns visuals
# # # # # #         if numeric_cols:
# # # # # #             st.markdown("**Numeric Columns Visualizations**")
# # # # # #             for col in numeric_cols:
# # # # # #                 st.markdown(f"### {col}")
# # # # # #                 col_distribution = df[col].dropna()
# # # # # #                 if len(col_distribution) > 0:
# # # # # #                     # Histogram for distribution
# # # # # #                     fig_hist = px.histogram(col_distribution, nbins=30, title=f"Histogram of {col}")
# # # # # #                     st.plotly_chart(fig_hist, use_container_width=True)
# # # # # #                     # Boxplot for outlier display
# # # # # #                     fig_box = px.box(df, y=col, title=f"Boxplot of {col}")
# # # # # #                     st.plotly_chart(fig_box, use_container_width=True)
        
# # # # # #         # Categorical columns visuals
# # # # # #         if categorical_cols:
# # # # # #             st.markdown("**Categorical Columns Visualizations**")
# # # # # #             for col in categorical_cols:
# # # # # #                 st.markdown(f"### {col}")
# # # # # #                 val_counts = df[col].value_counts()
# # # # # #                 if len(val_counts) > 0:
# # # # # #                     # Bar chart for category counts
# # # # # #                     fig_bar = px.bar(val_counts, x=val_counts.index, y=val_counts.values, 
# # # # # #                                      labels={"x": col, "y": "Count"},
# # # # # #                                      title=f"Bar Chart of {col}")
# # # # # #                     st.plotly_chart(fig_bar, use_container_width=True)
        
# # # # # #         # User-selected visualization
# # # # # #         st.subheader("Customizable Visualization")
# # # # # #         chosen_col = st.selectbox("Choose a column", df.columns)
# # # # # #         detected_type = ("Categorical" if chosen_col in categorical_cols else 
# # # # # #                          "Numeric" if chosen_col in numeric_cols else 
# # # # # #                          "Other / Unsupported")

# # # # # #         if detected_type == "Numeric":
# # # # # #             chart_type = st.selectbox("Select chart type", ["Histogram", "Boxplot", "Line Chart", "Scatter Plot with another Numeric Column"])
# # # # # #         elif detected_type == "Categorical":
# # # # # #             chart_type = st.selectbox("Select chart type", ["Bar Chart", "Pie Chart"])
# # # # # #         else:
# # # # # #             chart_type = None
# # # # # #             st.info("Unsupported data type selected.")

# # # # # #         if chart_type:
# # # # # #             if chart_type == "Histogram":
# # # # # #                 fig = px.histogram(df, x=chosen_col, title=f"Histogram of {chosen_col}")
# # # # # #                 st.plotly_chart(fig, use_container_width=True)
# # # # # #             elif chart_type == "Boxplot":
# # # # # #                 fig = px.box(df, y=chosen_col, title=f"Boxplot of {chosen_col}")
# # # # # #                 st.plotly_chart(fig, use_container_width=True)
# # # # # #             elif chart_type == "Line Chart":
# # # # # #                 fig = px.line(df, y=chosen_col, title=f"Line Chart of {chosen_col}")
# # # # # #                 st.plotly_chart(fig, use_container_width=True)
# # # # # #             elif chart_type == "Scatter Plot with another Numeric Column":
# # # # # #                 other_numeric_cols = [c for c in numeric_cols if c != chosen_col]
# # # # # #                 if other_numeric_cols:
# # # # # #                     col2 = st.selectbox("Select second numeric column for Scatter Plot", other_numeric_cols)
# # # # # #                     fig = px.scatter(df, x=chosen_col, y=col2, title=f"Scatter Plot of {chosen_col} vs {col2}")
# # # # # #                     st.plotly_chart(fig, use_container_width=True)
# # # # # #                 else:
# # # # # #                     st.info("Not enough numeric columns for scatter plot.")
# # # # # #             elif chart_type == "Bar Chart":
# # # # # #                 counts = df[chosen_col].value_counts()
# # # # # #                 fig = px.bar(counts, x=counts.index, y=counts.values, title=f"Bar Chart of {chosen_col}")
# # # # # #                 st.plotly_chart(fig, use_container_width=True)
# # # # # #             elif chart_type == "Pie Chart":
# # # # # #                 fig = px.pie(df, names=chosen_col, title=f"Pie Chart of {chosen_col}")
# # # # # #                 st.plotly_chart(fig, use_container_width=True)

# # # # # # # --- 4. Insights & Report ---
# # # # # # elif section == "Insights & Report":
# # # # # #     st.header("4️⃣ Data Insights & Report")
# # # # # #     if st.session_state.df_clean is None:
# # # # # #         st.warning("Please upload and preprocess data first.")
# # # # # #     else:
# # # # # #         df = st.session_state.df_clean

# # # # # #         st.subheader("Null Value Check")
# # # # # #         null_sum = df.isnull().sum()
# # # # # #         st.dataframe(null_sum[null_sum > 0].sort_values(ascending=False) if null_sum.sum() > 0 else "No missing values detected.")
        
# # # # # #         st.subheader("Outlier Detection (IQR method on numeric columns)")
# # # # # #         numeric_cols = df.select_dtypes(include=[np.number]).columns
# # # # # #         outlier_summary = {}
# # # # # #         for col in numeric_cols:
# # # # # #             outliers = detect_outliers_iqr(df[col].dropna())
# # # # # #             outlier_summary[col] = len(outliers)
# # # # # #         outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier Count'])
# # # # # #         st.dataframe(outlier_df)

# # # # # #         st.subheader("Skewness Detection (Numeric Columns)")
# # # # # #         skewness = calc_skewness_desc(df)
# # # # # #         st.dataframe(skewness)

# # # # # #         st.subheader("Cardinality Check")
# # # # # #         cardinality = cardinality_check(df)
# # # # # #         st.dataframe(cardinality)

# # # # # #         st.subheader("Correlation Issues (Absolute Correlation > 0.7)")
# # # # # #         high_corr = correlation_issues(df)
# # # # # #         if high_corr:
# # # # # #             corr_df = pd.DataFrame(high_corr, columns=["Column 1", "Column 2", "Correlation"])
# # # # # #             st.dataframe(corr_df)
# # # # # #         else:
# # # # # #             st.info("No strong correlation issues detected.")
        
# # # # # #         # Summary text report
# # # # # #         insights_text = io.StringIO()
# # # # # #         insights_text.write("Explorica Data Insights Summary\n")
# # # # # #         insights_text.write("===============================\n\n")
# # # # # #         insights_text.write("Null Values per Column:\n")
# # # # # #         for col, val in null_sum[null_sum > 0].items():
# # # # # #             insights_text.write(f"{col}: {val}\n")
# # # # # #         insights_text.write("\nOutlier Counts per Numeric Column:\n")
# # # # # #         for col, val in outlier_summary.items():
# # # # # #             insights_text.write(f"{col}: {val}\n")
# # # # # #         insights_text.write("\nSkewness per Numeric Column:\n")
# # # # # #         for col, val in skewness.dropna().items():
# # # # # #             insights_text.write(f"{col}: {val:.4f}\n")
# # # # # #         insights_text.write("\nCardinality per Column:\n")
# # # # # #         for col, val in cardinality.items():
# # # # # #             insights_text.write(f"{col}: {val}\n")
# # # # # #         if high_corr:
# # # # # #             insights_text.write("\nHighly Correlated Column Pairs (>0.7):\n")
# # # # # #             for c1, c2, corr_val in high_corr:
# # # # # #                 insights_text.write(f"{c1} & {c2}: {corr_val:.4f}\n")
# # # # # #         else:
# # # # # #             insights_text.write("\nNo high correlation pairs detected.\n")

# # # # # #         # Display full profiling report (optional)
# # # # # #         st.subheader("Profiling Report")
# # # # # #         profile = ProfileReport(df, title="Explorica Profiling Report", explorative=True)
# # # # # #         st_profile_report(profile)

# # # # # #         # Download cleaned data
# # # # # #         cleaned_csv = df.to_csv(index=False).encode('utf-8')
# # # # # #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

# # # # # #         # Download insights text
# # # # # #         st.download_button("📥 Download Insights Summary (TXT)", insights_text.getvalue(), "data_insights.txt", "text/plain")

# # # # # #         # Functionality to download one or more visualization plots can be added (example below):
# # # # # #         st.subheader("Download Visualization Example")

# # # # # #         # Example: Download correlation heatmap as PNG (using matplotlib and seaborn)
# # # # # #         fig, ax = plt.subplots(figsize=(8, 6))
# # # # # #         sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# # # # # #         plt.title("Correlation Heatmap")
# # # # # #         buf = io.BytesIO()
# # # # # #         plt.savefig(buf, format="png")
# # # # # #         buf.seek(0)
# # # # # #         st.download_button("📥 Download Correlation Heatmap", buf, "correlation_heatmap.png", "image/png")
# # # # # #         plt.close(fig)




# # # # # import streamlit as st
# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # import plotly.express as px
# # # # # from ydata_profiling import ProfileReport
# # # # # from streamlit_pandas_profiling import st_profile_report
# # # # # import io
# # # # # import seaborn as sns
# # # # # import matplotlib.pyplot as plt

# # # # # st.set_page_config(page_title="Explorica - Interactive Smart EDA", layout="wide")

# # # # # st.title("📊 Explorica - Interactive Smart Exploratory Data Analysis Dashboard")

# # # # # # Initialize session state for original, cleaned data, and cached reports
# # # # # if "df" not in st.session_state:
# # # # #     st.session_state.df = None
# # # # # if "df_clean" not in st.session_state:
# # # # #     st.session_state.df_clean = None
# # # # # if "profile_report_html" not in st.session_state:
# # # # #     st.session_state.profile_report_html = None

# # # # # # Helper functions
# # # # # def detect_outliers_iqr(data):
# # # # #     Q1 = np.percentile(data, 25)
# # # # #     Q3 = np.percentile(data, 75)
# # # # #     IQR = Q3 - Q1
# # # # #     lower_bound = Q1 - 1.5 * IQR
# # # # #     upper_bound = Q3 + 1.5 * IQR
# # # # #     return np.where((data < lower_bound) | (data > upper_bound))[0]

# # # # # def calc_skewness(df):
# # # # #     return df.skew(numeric_only=True)

# # # # # def cardinality_check(df):
# # # # #     return df.nunique()

# # # # # def correlation_issues(df, threshold=0.7):
# # # # #     numeric_df = df.select_dtypes(include=[np.number])
# # # # # corr = numeric_df.corr().abs()
# # # # # issues = []
# # # # # for i in range(len(corr.columns)):
# # # # #         for j in range(i+1, len(corr.columns)):
# # # # #             if corr.iloc[i, j] > threshold:
# # # # #                 issues.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
# # # # #                 return issues

# # # # # # Use tabs for better UX
# # # # # tab_upload, tab_preprocess, tab_visualize, tab_insights = st.tabs(
# # # # #     ["Upload Data 🗂️", "Preprocess 🧹", "Visualize 📊", "Insights & Report 📈"]
# # # # # )

# # # # # with tab_upload:
# # # # #     st.header("1️⃣ Upload your CSV file")
# # # # #     uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])
# # # # #     if uploaded_file is not None:
# # # # #         df = pd.read_csv(uploaded_file)
# # # # #         st.session_state.df = df
# # # # #         st.session_state.df_clean = df.copy()
# # # # #         st.session_state.profile_report_html = None
# # # # #         st.success("✅ Data uploaded successfully!")
# # # # #         with st.expander("Show Data Preview & Summary"):
# # # # #             st.dataframe(df.head())
# # # # #             st.markdown(f"**Data Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# # # # #             # Detect and show column data types
# # # # #             dtypes = df.dtypes.astype(str)
# # # # #             col_types = []
# # # # #             for col in df.columns:
# # # # #                 if pd.api.types.is_numeric_dtype(df[col]):
# # # # #                     col_types.append("Numeric")
# # # # #                 elif pd.api.types.is_datetime64_any_dtype(df[col]):
# # # # #                     col_types.append("Datetime")
# # # # #                 elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
# # # # #                     col_types.append("Categorical")
# # # # #                 else:
# # # # #                     col_types.append("Other")
# # # # #             type_df = pd.DataFrame({"Column": df.columns, "Data Type": dtypes.values, "Detected Type": col_types})
# # # # #             st.dataframe(type_df)

# # # # #             st.subheader("Summary Statistics")
# # # # #             st.dataframe(df.describe(include='all').transpose())

# # # # # with tab_preprocess:
# # # # #     st.header("2️⃣ Data Preprocessing")
# # # # #     if st.session_state.df is None:
# # # # #         st.warning("Please upload data first in the 'Upload Data' tab.")
# # # # #     else:
# # # # #         df = st.session_state.df_clean.copy()

# # # # #         st.subheader("Missing and Empty Values Summary")
# # # # #         missing = df.isnull().sum()
# # # # #         empty_str = (df == '').sum()
# # # # #         missing_df = pd.DataFrame({"Missing Values": missing, "Empty String Values": empty_str})
# # # # #         missing_df = missing_df[missing_df.sum(axis=1) > 0]  # only rows with missing or empty
# # # # #         if missing_df.empty:
# # # # #             st.info("No missing or empty string values detected.")
# # # # #         else:
# # # # #             st.dataframe(missing_df)

# # # # #             st.subheader("Fill Missing Values")
# # # # #             fill_option = st.selectbox(
# # # # #                 "Choose method to handle missing values:",
# # # # #                 ["None", "Drop rows with missing values",
# # # # #                  "Fill with Mean (numeric columns only)",
# # # # #                  "Fill with Median (numeric columns only)",
# # # # #                  "Fill with Mode (all columns)", "Fill with Custom Values"]
# # # # #             )

# # # # #             custom_values = {}
# # # # #             if fill_option == "Fill with Custom Values":
# # # # #                 for col in missing_df.index:
# # # # #                     val = st.text_input(f"Custom fill value for column '{col}': (leave blank to skip)", key=f"custom_{col}")
# # # # #                     custom_values[col] = val

# # # # #             if st.button("Apply missing value handling"):
# # # # #                 if fill_option == "None":
# # # # #                     st.info("No changes made to missing values.")
# # # # #                 elif fill_option == "Drop rows with missing values":
# # # # #                     before_rows = df.shape[0]
# # # # #                     df.dropna(inplace=True)
# # # # #                     after_rows = df.shape[0]
# # # # #                     st.success(f"Dropped {before_rows - after_rows} rows containing missing values.")
# # # # #                 elif fill_option == "Fill with Mean (numeric columns only)":
# # # # #                     for col in df.select_dtypes(include=np.number).columns:
# # # # #                         if col in missing_df.index:
# # # # #                             df[col].fillna(df[col].mean(), inplace=True)
# # # # #                     st.success("Filled missing numeric values with mean.")
# # # # #                 elif fill_option == "Fill with Median (numeric columns only)":
# # # # #                     for col in df.select_dtypes(include=np.number).columns:
# # # # #                         if col in missing_df.index:
# # # # #                             df[col].fillna(df[col].median(), inplace=True)
# # # # #                     st.success("Filled missing numeric values with median.")
# # # # #                 elif fill_option == "Fill with Mode (all columns)":
# # # # #                     for col in missing_df.index:
# # # # #                         mode_val = df[col].mode()
# # # # #                         if not mode_val.empty:
# # # # #                             df[col].fillna(mode_val[0], inplace=True)
# # # # #                     st.success("Filled missing values with mode.")
# # # # #                 elif fill_option == "Fill with Custom Values":
# # # # #                     for col, val in custom_values.items():
# # # # #                         if val != "":
# # # # #                             try:
# # # # #                                 dtype = df[col].dtype
# # # # #                                 if pd.api.types.is_numeric_dtype(dtype):
# # # # #                                     val_conv = float(val)
# # # # #                                 else:
# # # # #                                     val_conv = val
# # # # #                                 df[col].fillna(val_conv, inplace=True)
# # # # #                             except Exception:
# # # # #                                 st.warning(f"Invalid input for column {col}. Skipping.")
# # # # #                     st.success("Applied custom missing value fills.")

# # # # #                 st.session_state.df_clean = df

# # # # #         st.subheader("Preview of Preprocessed Data")
# # # # #         st.dataframe(df.head())

# # # # #         cleaned_csv = df.to_csv(index=False).encode('utf-8')
# # # # #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

# # # # # with tab_visualize:
# # # # #     st.header("3️⃣ Data Visualization")
# # # # #     if st.session_state.df_clean is None:
# # # # #         st.warning("Please upload and preprocess data first.")
# # # # #     else:
# # # # #         df = st.session_state.df_clean.copy()
# # # # #         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# # # # #         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# # # # #         st.subheader("Auto Visualizations")

# # # # #         # Layout two columns for side-by-side visuals
# # # # #         col1, col2 = st.columns(2)

# # # # #         with col1:
# # # # #             if numeric_cols:
# # # # #                 st.markdown("### Numeric Data Distributions")
# # # # #                 for col_name in numeric_cols:
# # # # #                     fig_hist = px.histogram(df, x=col_name, nbins=30, title=f"Histogram of {col_name}", marginal="rug", hover_data=df.columns)
# # # # #                     st.plotly_chart(fig_hist, use_container_width=True)
# # # # #                     fig_box = px.box(df, y=col_name, points="outliers", title=f"Boxplot of {col_name}")
# # # # #                     st.plotly_chart(fig_box, use_container_width=True)

# # # # #         with col2:
# # # # #             if categorical_cols:
# # # # #                 st.markdown("### Categorical Data Counts")
# # # # #                 for col_name in categorical_cols:
# # # # #                     counts = df[col_name].value_counts().reset_index()
# # # # #                     counts.columns = [col_name, 'Count']
# # # # #                     fig_bar = px.bar(counts, x=col_name, y='Count', title=f"Bar chart of {col_name}")
# # # # #                     st.plotly_chart(fig_bar, use_container_width=True)

# # # # #         # Additional automatic multivariate plots for numeric data
# # # # #         if len(numeric_cols) > 1:
# # # # #             st.subheader("Multivariate Numeric Visualizations")
# # # # #             fig_scatter = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix - Numeric Columns")
# # # # #             st.plotly_chart(fig_scatter, use_container_width=True)
# # # # #             corr = df[numeric_cols].corr()
# # # # #             fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
# # # # #             st.plotly_chart(fig_heatmap, use_container_width=True)

# # # # #         # Custom Visualization Area
# # # # #         st.subheader("Custom Visualization")
# # # # #         sel_col = st.selectbox("Select column to visualize", df.columns)

# # # # #         # Detect type
# # # # #         if sel_col in numeric_cols:
# # # # #             sel_chart = st.selectbox("Select chart type", ["Histogram", "Boxplot", "Line Chart", "Scatter Plot (select another column)"])
# # # # #         elif sel_col in categorical_cols:
# # # # #             sel_chart = st.selectbox("Select chart type", ["Bar Chart", "Pie Chart"])
# # # # #         else:
# # # # #             sel_chart = None
# # # # #             st.info("Selected column not supported for custom visualization.")

# # # # #         if sel_chart and sel_col:
# # # # #             if sel_chart == "Histogram":
# # # # #                 fig = px.histogram(df, x=sel_col, nbins=30, title=f"Histogram of {sel_col}")
# # # # #                 st.plotly_chart(fig, use_container_width=True)
# # # # #             elif sel_chart == "Boxplot":
# # # # #                 fig = px.box(df, y=sel_col, title=f"Boxplot of {sel_col}")
# # # # #                 st.plotly_chart(fig, use_container_width=True)
# # # # #             elif sel_chart == "Line Chart":
# # # # #                 fig = px.line(df, y=sel_col, title=f"Line Chart of {sel_col}")
# # # # #                 st.plotly_chart(fig, use_container_width=True)
# # # # #             elif sel_chart == "Scatter Plot (select another column)":
# # # # #                 other_numeric_cols = [c for c in numeric_cols if c != sel_col]
# # # # #                 if other_numeric_cols:
# # # # #                     x_col = st.selectbox("Select column for X-axis", other_numeric_cols)
# # # # #                     fig = px.scatter(df, x=x_col, y=sel_col, title=f"Scatter Plot: {sel_col} vs {x_col}")
# # # # #                     st.plotly_chart(fig, use_container_width=True)
# # # # #                 else:
# # # # #                     st.info("Need at least two numeric columns for scatter plot.")
# # # # #             elif sel_chart == "Bar Chart":
# # # # #                 counts = df[sel_col].value_counts().reset_index()
# # # # #                 counts.columns = [sel_col, "Count"]
# # # # #                 fig = px.bar(counts, x=sel_col, y="Count", title=f"Bar Chart of {sel_col}")
# # # # #                 st.plotly_chart(fig, use_container_width=True)
# # # # #             elif sel_chart == "Pie Chart":
# # # # #                 fig = px.pie(df, names=sel_col, title=f"Pie Chart of {sel_col}")
# # # # #                 st.plotly_chart(fig, use_container_width=True)

# # # # # with tab_insights:
# # # # #     st.header("4️⃣ Data Insights & Report")
# # # # #     if st.session_state.df_clean is None:
# # # # #         st.warning("Please upload and preprocess data first.")
# # # # #     else:
# # # # #         df = st.session_state.df_clean.copy()

# # # # #         st.subheader("Null Value Overview")
# # # # #         null_sum = df.isnull().sum()
# # # # #         if null_sum.sum() > 0:
# # # # #             st.dataframe(null_sum[null_sum > 0].sort_values(ascending=False))
# # # # #         else:
# # # # #             st.info("No missing values detected.")

# # # # #         st.subheader("Outliers Detection using IQR")
# # # # #         numeric_cols = df.select_dtypes(include=np.number).columns
# # # # #         outlier_counts = {}
# # # # #         for col in numeric_cols:
# # # # #             outlier_idx = detect_outliers_iqr(df[col].dropna())
# # # # #             outlier_counts[col] = len(outlier_idx)
# # # # #         outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=["Outlier Count"])
# # # # #         st.dataframe(outlier_df)

# # # # #         st.subheader("Skewness of Numeric Features")
# # # # #         skewness = calc_skewness(df)
# # # # #         st.dataframe(skewness)

# # # # #         st.subheader("Cardinality Check")
# # # # #         cardinalities = cardinality_check(df)
# # # # #         st.dataframe(cardinalities)

# # # # #         st.subheader("Strong Correlation (>|0.7|) Pairs")
# # # # #         corr_issues = correlation_issues(df)
# # # # #         if corr_issues:
# # # # #             corr_df = pd.DataFrame(corr_issues, columns=["Feature 1", "Feature 2", "Correlation"])
# # # # #             st.dataframe(corr_df)
# # # # #         else:
# # # # #             st.info("No strong correlations detected.")

# # # # #         # Generate insights summary text for download
# # # # #         insights_str = io.StringIO()
# # # # #         insights_str.write("Explorica Data Insights Summary\n")
# # # # #         insights_str.write("===============================\n\n")
# # # # #         insights_str.write("Null Values:\n")
# # # # #         for col, val in null_sum[null_sum > 0].items():
# # # # #             insights_str.write(f"{col}: {val}\n")
# # # # #         insights_str.write("\nOutlier Counts (IQR) per Numeric Column:\n")
# # # # #         for col, val in outlier_counts.items():
# # # # #             insights_str.write(f"{col}: {val}\n")
# # # # #         insights_str.write("\nSkewness of Numeric Columns:\n")
# # # # #         for col, val in skewness.dropna().items():
# # # # #             insights_str.write(f"{col}: {val:.4f}\n")
# # # # #         insights_str.write("\nCardinality per Column:\n")
# # # # #         for col, val in cardinalities.items():
# # # # #             insights_str.write(f"{col}: {val}\n")
# # # # #         if corr_issues:
# # # # #             insights_str.write("\nStrongly Correlated Pairs (>|0.7|):\n")
# # # # #             for c1, c2, corr_val in corr_issues:
# # # # #                 insights_str.write(f"{c1} & {c2}: {corr_val:.4f}\n")
# # # # #         else:
# # # # #             insights_str.write("\nNo strong correlations detected.\n")

# # # # #         # Display profiling report with caching
# # # # #         st.subheader("Profiling Report")
# # # # #         if st.session_state.profile_report_html is None:
# # # # #             with st.spinner("Generating profiling report..."):
# # # # #                 profile = ProfileReport(df, title="Explorica Profiler", explorative=True)
# # # # #                 st.session_state.profile_report_html = profile.to_html()
# # # # #                 st_profile_report(profile)
# # # # #         else:
# # # # #             st.markdown("Profiling report loaded from cache.")
# # # # #             st_profile_report(ProfileReport(df, title="Explorica Profiler", explorative=True))

# # # # #         # Download cleaned CSV
# # # # #         cleaned_csv = df.to_csv(index=False).encode('utf-8')
# # # # #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

# # # # #         # Download insights summary text
# # # # #         st.download_button("📥 Download Insights Summary (TXT)", insights_str.getvalue(), "insights_summary.txt", "text/plain")

# # # # #         # Download example visualization: Correlation Heatmap as PNG
# # # # #         st.subheader("Download Example Visualization: Correlation Heatmap")

# # # # #         fig, ax = plt.subplots(figsize=(8, 6))
# # # # #         sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# # # # #         plt.title("Correlation Heatmap")
# # # # #         buf = io.BytesIO()
# # # # #         plt.savefig(buf, format="png")
# # # # #         buf.seek(0)
# # # # #         st.download_button("📥 Download Correlation Heatmap", buf, "correlation_heatmap.png", "image/png")
# # # # #         plt.close(fig)
# # # # import streamlit as st
# # # # import pandas as pd
# # # # import numpy as np
# # # # import plotly.express as px
# # # # from ydata_profiling import ProfileReport
# # # # from streamlit_pandas_profiling import st_profile_report
# # # # import io
# # # # import seaborn as sns
# # # # import matplotlib.pyplot as plt

# # # # st.set_page_config(page_title="Explorica - Interactive Smart EDA", layout="wide")

# # # # st.title("📊 Explorica - Interactive Smart Exploratory Data Analysis Dashboard")

# # # # # Initialize session state for original, cleaned data, and cached reports
# # # # if "df" not in st.session_state:
# # # #     st.session_state.df = None
# # # # if "df_clean" not in st.session_state:
# # # #     st.session_state.df_clean = None
# # # # if "profile_report_html" not in st.session_state:
# # # #     st.session_state.profile_report_html = None

# # # # # Helper functions
# # # # def detect_outliers_iqr(data):
# # # #     Q1 = np.percentile(data, 25)
# # # #     Q3 = np.percentile(data, 75)
# # # #     IQR = Q3 - Q1
# # # #     lower_bound = Q1 - 1.5 * IQR
# # # #     upper_bound = Q3 + 1.5 * IQR
# # # #     return np.where((data < lower_bound) | (data > upper_bound))[0]

# # # # def calc_skewness(df):
# # # #     return df.skew(numeric_only=True)

# # # # def cardinality_check(df):
# # # #     return df.nunique()

# # # # def correlation_issues(df, threshold=0.7):
# # # #     numeric_df = df.select_dtypes(include=[np.number])
# # # #     corr = numeric_df.corr().abs()
# # # #     issues = []
# # # #     for i in range(len(corr.columns)):
# # # #         for j in range(i+1, len(corr.columns)):
# # # #             if corr.iloc[i, j] > threshold:
# # # #                 issues.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
# # # #     return issues

# # # # # Use tabs for better UX
# # # # tab_upload, tab_preprocess, tab_visualize, tab_insights = st.tabs(
# # # #     ["Upload Data 🗂️", "Preprocess 🧹", "Visualize 📊", "Insights & Report 📈"]
# # # # )

# # # # with tab_upload:
# # # #     st.header("1️⃣ Upload your CSV file")
# # # #     uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])
# # # #     if uploaded_file is not None:
# # # #         df = pd.read_csv(uploaded_file)
# # # #         st.session_state.df = df
# # # #         st.session_state.df_clean = df.copy()
# # # #         st.session_state.profile_report_html = None
# # # #         st.success("✅ Data uploaded successfully!")
# # # #         with st.expander("Show Data Preview & Summary"):
# # # #             st.dataframe(df.head())
# # # #             st.markdown(f"**Data Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# # # #             # Detect and show column data types
# # # #             dtypes = df.dtypes.astype(str)
# # # #             col_types = []
# # # #             for col in df.columns:
# # # #                 if pd.api.types.is_numeric_dtype(df[col]):
# # # #                     col_types.append("Numeric")
# # # #                 elif pd.api.types.is_datetime64_any_dtype(df[col]):
# # # #                     col_types.append("Datetime")
# # # #                 elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
# # # #                     col_types.append("Categorical")
# # # #                 else:
# # # #                     col_types.append("Other")
# # # #             type_df = pd.DataFrame({"Column": df.columns, "Data Type": dtypes.values, "Detected Type": col_types})
# # # #             st.dataframe(type_df)

# # # #             st.subheader("Summary Statistics")
# # # #             st.dataframe(df.describe(include='all').transpose())

# # # # with tab_preprocess:
# # # #     st.header("2️⃣ Data Preprocessing")
# # # #     if st.session_state.df is None:
# # # #         st.warning("Please upload data first in the 'Upload Data' tab.")
# # # #     else:
# # # #         df = st.session_state.df_clean.copy()

# # # #         st.subheader("Missing and Empty Values Summary")
# # # #         missing = df.isnull().sum()
# # # #         empty_str = (df == '').sum()
# # # #         missing_df = pd.DataFrame({"Missing Values": missing, "Empty String Values": empty_str})
# # # #         missing_df = missing_df[missing_df.sum(axis=1) > 0]  # only rows with missing or empty
# # # #         if missing_df.empty:
# # # #             st.info("No missing or empty string values detected.")
# # # #         else:
# # # #             st.dataframe(missing_df)

# # # #             st.subheader("Fill Missing Values")
# # # #             fill_option = st.selectbox(
# # # #                 "Choose method to handle missing values:",
# # # #                 ["None",
# # # #                  "Drop rows with missing values",
# # # #                  "Fill with Mean (numeric columns only)",
# # # #                  "Fill with Median (numeric columns only)",
# # # #                  "Fill with Mode (all columns)",
# # # #                  "Fill with Custom Values"]
# # # #             )

# # # #             custom_values = {}
# # # #             if fill_option == "Fill with Custom Values":
# # # #                 for col in missing_df.index:
# # # #                     val = st.text_input(f"Custom fill value for column '{col}': (leave blank to skip)", key=f"custom_{col}")
# # # #                     custom_values[col] = val

# # # #             if st.button("Apply missing value handling", key="apply_missing"):
# # # #                 if fill_option == "None":
# # # #                     st.info("No changes made to missing values.")
# # # #                 elif fill_option == "Drop rows with missing values":
# # # #                     before_rows = df.shape[0]
# # # #                     df.dropna(inplace=True)
# # # #                     after_rows = df.shape[0]
# # # #                     st.success(f"Dropped {before_rows - after_rows} rows containing missing values.")
# # # #                 elif fill_option == "Fill with Mean (numeric columns only)":
# # # #                     for col in df.select_dtypes(include=np.number).columns:
# # # #                         if col in missing_df.index:
# # # #                             df[col].fillna(df[col].mean(), inplace=True)
# # # #                     st.success("Filled missing numeric values with mean.")
# # # #                 elif fill_option == "Fill with Median (numeric columns only)":
# # # #                     for col in df.select_dtypes(include=np.number).columns:
# # # #                         if col in missing_df.index:
# # # #                             df[col].fillna(df[col].median(), inplace=True)
# # # #                     st.success("Filled missing numeric values with median.")
# # # #                 elif fill_option == "Fill with Mode (all columns)":
# # # #                     for col in missing_df.index:
# # # #                         mode_val = df[col].mode()
# # # #                         if not mode_val.empty:
# # # #                             df[col].fillna(mode_val[0], inplace=True)
# # # #                     st.success("Filled missing values with mode.")
# # # #                 elif fill_option == "Fill with Custom Values":
# # # #                     for col, val in custom_values.items():
# # # #                         if val != "":
# # # #                             try:
# # # #                                 dtype = df[col].dtype
# # # #                                 if pd.api.types.is_numeric_dtype(dtype):
# # # #                                     val_conv = float(val)
# # # #                                 else:
# # # #                                     val_conv = val
# # # #                                 df[col].fillna(val_conv, inplace=True)
# # # #                             except Exception:
# # # #                                 st.warning(f"Invalid input for column {col}. Skipping.")
# # # #                     st.success("Applied custom missing value fills.")

# # # #                 st.session_state.df_clean = df

# # # #         st.subheader("Preview of Preprocessed Data")
# # # #         st.dataframe(df.head())

# # # #         cleaned_csv = df.to_csv(index=False).encode('utf-8')
# # # #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv")

# # # # with tab_visualize:
# # # #     st.header("3️⃣ Data Visualization")
# # # #     if st.session_state.df_clean is None:
# # # #         st.warning("Please upload and preprocess data first.")
# # # #     else:
# # # #         df = st.session_state.df_clean.copy()
# # # #         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# # # #         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# # # #         st.subheader("Auto Visualizations")

# # # #         # Layout two columns for side-by-side visuals
# # # #         col1, col2 = st.columns(2)

# # # #         with col1:
# # # #             if numeric_cols:
# # # #                 st.markdown("### Numeric Data Distributions")
# # # #                 for col_name in numeric_cols:
# # # #                     fig_hist = px.histogram(df, x=col_name, nbins=30, title=f"Histogram of {col_name}", marginal="rug", hover_data=df.columns)
# # # #                     st.plotly_chart(fig_hist, use_container_width=True)
# # # #                     fig_box = px.box(df, y=col_name, points="outliers", title=f"Boxplot of {col_name}")
# # # #                     st.plotly_chart(fig_box, use_container_width=True)

# # # #         with col2:
# # # #             if categorical_cols:
# # # #                 st.markdown("### Categorical Data Counts")
# # # #                 for col_name in categorical_cols:
# # # #                     counts = df[col_name].value_counts().reset_index()
# # # #                     counts.columns = [col_name, 'Count']
# # # #                     fig_bar = px.bar(counts, x=col_name, y='Count', title=f"Bar chart of {col_name}")
# # # #                     st.plotly_chart(fig_bar, use_container_width=True)

# # # #         # Additional automatic multivariate plots for numeric data
# # # #         if len(numeric_cols) > 1:
# # # #             st.subheader("Multivariate Numeric Visualizations")
# # # #             fig_scatter = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix - Numeric Columns")
# # # #             st.plotly_chart(fig_scatter, use_container_width=True)
# # # #             corr = df[numeric_cols].corr()
# # # #             fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
# # # #             st.plotly_chart(fig_heatmap, use_container_width=True)

# # # #         # Custom Visualization Area
# # # #         st.subheader("Custom Visualization")
# # # #         sel_col = st.selectbox("Select column to visualize", df.columns)

# # # #         # Detect type
# # # #         if sel_col in numeric_cols:
# # # #             sel_chart = st.selectbox("Select chart type", ["Histogram", "Boxplot", "Line Chart", "Scatter Plot (select another column)"])
# # # #         elif sel_col in categorical_cols:
# # # #             sel_chart = st.selectbox("Select chart type", ["Bar Chart", "Pie Chart"])
# # # #         else:
# # # #             sel_chart = None
# # # #             st.info("Selected column not supported for custom visualization.")

# # # #         if sel_chart and sel_col:
# # # #             if sel_chart == "Histogram":
# # # #                 fig = px.histogram(df, x=sel_col, nbins=30, title=f"Histogram of {sel_col}")
# # # #                 st.plotly_chart(fig, use_container_width=True)
# # # #             elif sel_chart == "Boxplot":
# # # #                 fig = px.box(df, y=sel_col, title=f"Boxplot of {sel_col}")
# # # #                 st.plotly_chart(fig, use_container_width=True)
# # # #             elif sel_chart == "Line Chart":
# # # #                 fig = px.line(df, y=sel_col, title=f"Line Chart of {sel_col}")
# # # #                 st.plotly_chart(fig, use_container_width=True)
# # # #             elif sel_chart == "Scatter Plot (select another column)":
# # # #                 other_numeric_cols = [c for c in numeric_cols if c != sel_col]
# # # #                 if other_numeric_cols:
# # # #                     x_col = st.selectbox("Select column for X-axis", other_numeric_cols)
# # # #                     fig = px.scatter(df, x=x_col, y=sel_col, title=f"Scatter Plot: {sel_col} vs {x_col}")
# # # #                     st.plotly_chart(fig, use_container_width=True)
# # # #                 else:
# # # #                     st.info("Need at least two numeric columns for scatter plot.")
# # # #             elif sel_chart == "Bar Chart":
# # # #                 counts = df[sel_col].value_counts().reset_index()
# # # #                 counts.columns = [sel_col, "Count"]
# # # #                 fig = px.bar(counts, x=sel_col, y="Count", title=f"Bar Chart of {sel_col}")
# # # #                 st.plotly_chart(fig, use_container_width=True)
# # # #             elif sel_chart == "Pie Chart":
# # # #                 fig = px.pie(df, names=sel_col, title=f"Pie Chart of {sel_col}")
# # # #                 st.plotly_chart(fig, use_container_width=True)

# # # # with tab_insights:
# # # #     st.header("4️⃣ Data Insights & Report")
# # # #     if st.session_state.df_clean is None:
# # # #         st.warning("Please upload and preprocess data first.")
# # # #     else:
# # # #         df = st.session_state.df_clean.copy()

# # # #         st.subheader("Null Value Overview")
# # # #         null_sum = df.isnull().sum()
# # # #         if null_sum.sum() > 0:
# # # #             st.dataframe(null_sum[null_sum > 0].sort_values(ascending=False))
# # # #         else:
# # # #             st.info("No missing values detected.")

# # # #         st.subheader("Outliers Detection using IQR")
# # # #         numeric_cols = df.select_dtypes(include=np.number).columns
# # # #         outlier_counts = {}
# # # #         for col in numeric_cols:
# # # #             outlier_idx = detect_outliers_iqr(df[col].dropna())
# # # #             outlier_counts[col] = len(outlier_idx)
# # # #         outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=["Outlier Count"])
# # # #         st.dataframe(outlier_df)

# # # #         st.subheader("Skewness of Numeric Features")
# # # #         skewness = calc_skewness(df)
# # # #         st.dataframe(skewness)

# # # #         st.subheader("Cardinality Check")
# # # #         cardinalities = cardinality_check(df)
# # # #         st.dataframe(cardinalities)

# # # #         st.subheader("Strong Correlation (>|0.7|) Pairs")
# # # #         corr_issues = correlation_issues(df)
# # # #         if corr_issues:
# # # #             corr_df = pd.DataFrame(corr_issues, columns=["Feature 1", "Feature 2", "Correlation"])
# # # #             st.dataframe(corr_df)
# # # #         else:
# # # #             st.info("No strong correlations detected.")

# # # #         # Generate insights summary text for download
# # # #         insights_str = io.StringIO()
# # # #         insights_str.write("Explorica Data Insights Summary\n")
# # # #         insights_str.write("===============================\n\n")
# # # #         insights_str.write("Null Values:\n")
# # # #         for col, val in null_sum[null_sum > 0].items():
# # # #             insights_str.write(f"{col}: {val}\n")
# # # #         insights_str.write("\nOutlier Counts (IQR) per Numeric Column:\n")
# # # #         for col, val in outlier_counts.items():
# # # #             insights_str.write(f"{col}: {val}\n")
# # # #         insights_str.write("\nSkewness of Numeric Columns:\n")
# # # #         for col, val in skewness.dropna().items():
# # # #             insights_str.write(f"{col}: {val:.4f}\n")
# # # #         insights_str.write("\nCardinality per Column:\n")
# # # #         for col, val in cardinalities.items():
# # # #             insights_str.write(f"{col}: {val}\n")
# # # #         if corr_issues:
# # # #             insights_str.write("\nStrongly Correlated Pairs (>|0.7|):\n")
# # # #             for c1, c2, corr_val in corr_issues:
# # # #                 insights_str.write(f"{c1} & {c2}: {corr_val:.4f}\n")
# # # #         else:
# # # #             insights_str.write("\nNo strong correlations detected.\n")

# # # #         # Display profiling report with caching
# # # #         st.subheader("Profiling Report")
# # # #         if st.session_state.profile_report_html is None:
# # # #             with st.spinner("Generating profiling report..."):
# # # #                 profile = ProfileReport(df, title="Explorica Profiler", explorative=True)
# # # #                 st.session_state.profile_report_html = profile.to_html()
# # # #                 st_profile_report(profile)
# # # #         else:
# # # #             st.markdown("Profiling report loaded from cache.")
# # # #             st_profile_report(ProfileReport(df, title="Explorica Profiler", explorative=True))

# # # #         # Download cleaned CSV
# # # #         cleaned_csv = df.to_csv(index=False).encode('utf-8')
# # # #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv_report")

# # # #         # Download insights summary text
# # # #         st.download_button("📥 Download Insights Summary (TXT)", insights_str.getvalue(), "insights_summary.txt", "text/plain", key="download_insights_summary")

# # # #         # Download example visualization: Correlation Heatmap as PNG
# # # #         st.subheader("Download Example Visualization: Correlation Heatmap")

# # # #         fig, ax = plt.subplots(figsize=(8, 6))
# # # #         # sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# # # #         numeric_df = df.select_dtypes(include=[np.number])
# # # #         sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)

# # # #         plt.title("Correlation Heatmap")
# # # #         buf = io.BytesIO()
# # # #         plt.savefig(buf, format="png")
# # # #         buf.seek(0)
# # # #         st.download_button("📥 Download Correlation Heatmap", buf, "correlation_heatmap.png", "image/png", key="download_corr_heatmap")
# # # #         plt.close(fig)
# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import plotly.express as px
# # # from ydata_profiling import ProfileReport
# # # from streamlit_pandas_profiling import st_profile_report
# # # import io
# # # import seaborn as sns
# # # import matplotlib.pyplot as plt
# # # from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# # # st.set_page_config(page_title="Explorica - Enhanced Smart EDA", layout="wide")
# # # st.title("📊 Explorica - Enhanced Interactive Smart Exploratory Data Analysis Dashboard")

# # # # Initialize session state
# # # if "df" not in st.session_state:
# # #     st.session_state.df = None
# # # if "df_clean" not in st.session_state:
# # #     st.session_state.df_clean = None
# # # if "profile_report_html" not in st.session_state:
# # #     st.session_state.profile_report_html = None

# # # # Helper functions
# # # def detect_outliers_iqr(data):
# # #     Q1 = np.percentile(data, 25)
# # #     Q3 = np.percentile(data, 75)
# # #     IQR = Q3 - Q1
# # #     lower_bound = Q1 - 1.5 * IQR
# # #     upper_bound = Q3 + 1.5 * IQR
# # #     return np.where((data < lower_bound) | (data > upper_bound))[0]

# # # def calc_skewness(df):
# # #     return df.skew(numeric_only=True)

# # # def cardinality_check(df):
# # #     return df.nunique()

# # # def correlation_issues(df, threshold=0.7):
# # #     numeric_df = df.select_dtypes(include=[np.number])
# # #     corr = numeric_df.corr().abs()
# # #     issues = []
# # #     for i in range(len(corr.columns)):
# # #         for j in range(i+1, len(corr.columns)):
# # #             if corr.iloc[i, j] > threshold:
# # #                 issues.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
# # #     return issues

# # # def convert_column_types(df):
# # #     suggestions = {}
# # #     for col in df.columns:
# # #         if str(df[col].dtype) == 'object':
# # #             # Try to convert to datetime
# # #             try:
# # #                 pd.to_datetime(df[col], errors='raise')
# # #                 suggestions[col] = 'datetime'
# # #             except:
# # #                 # Suggest categorical if unique values are low relative to length
# # #                 if df[col].nunique() / len(df) < 0.05:
# # #                     suggestions[col] = 'categorical'
# # #     return suggestions

# # # def scale_numeric_columns(df, method='StandardScaler'):
# # #     numeric_cols = df.select_dtypes(include=[np.number]).columns
# # #     if method == 'StandardScaler':
# # #         scaler = StandardScaler()
# # #     elif method == 'MinMaxScaler':
# # #         scaler = MinMaxScaler()
# # #     else:
# # #         return df
# # #     df_scaled = df.copy()
# # #     df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
# # #     return df_scaled

# # # def handle_outliers_iqr(df, cols, method='cap'):
# # #     df_out = df.copy()
# # #     for col in cols:
# # #         data = df_out[col].dropna()
# # #         outlier_idx = detect_outliers_iqr(data)
# # #         Q1 = np.percentile(data, 25)
# # #         Q3 = np.percentile(data, 75)
# # #         IQR = Q3 - Q1
# # #         lower_bound = Q1 - 1.5 * IQR
# # #         upper_bound = Q3 + 1.5 * IQR
# # #         if method == 'cap':
# # #             df_out.loc[df_out[col] < lower_bound, col] = lower_bound
# # #             df_out.loc[df_out[col] > upper_bound, col] = upper_bound
# # #         elif method == 'remove':
# # #             df_out = df_out.drop(df_out[(df_out[col] < lower_bound) | (df_out[col] > upper_bound)].index)
# # #     return df_out

# # # def label_encode_columns(df, cols):
# # #     df_enc = df.copy()
# # #     le = LabelEncoder()
# # #     for col in cols:
# # #         df_enc[col] = le.fit_transform(df_enc[col].astype(str))
# # #     return df_enc

# # # # Tabs for navigation
# # # tab_upload, tab_preprocess, tab_visualize, tab_insights = st.tabs(
# # #     ["Upload Data 🗂️", "Preprocess 🧹", "Visualize 📊", "Insights & Report 📈"]
# # # )

# # # with tab_upload:
# # #     st.header("1️⃣ Upload your CSV file")
# # #     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
# # #     if uploaded_file is not None:
# # #         df = pd.read_csv(uploaded_file)
# # #         st.session_state.df = df
# # #         st.session_state.df_clean = df.copy()
# # #         st.session_state.profile_report_html = None
# # #         st.success("✅ Data uploaded successfully!")

# # #         with st.expander("Show Data Preview & Summary"):
# # #             st.dataframe(df.head())
# # #             st.markdown(f"**Data Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# # #             # Detect and show column data types
# # #             dtypes = df.dtypes.astype(str)
# # #             col_types = []
# # #             for col in df.columns:
# # #                 if pd.api.types.is_numeric_dtype(df[col]):
# # #                     col_types.append("Numeric")
# # #                 elif pd.api.types.is_datetime64_any_dtype(df[col]):
# # #                     col_types.append("Datetime")
# # #                 elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
# # #                     col_types.append("Categorical")
# # #                 else:
# # #                     col_types.append("Other")
# # #             type_df = pd.DataFrame({"Column": df.columns, "Data Type": dtypes.values, "Detected Type": col_types})
# # #             st.dataframe(type_df)

# # #             st.subheader("Summary Statistics")
# # #             st.dataframe(df.describe(include='all').transpose())

# # # with tab_preprocess:
# # #     st.header("2️⃣ Data Preprocessing")
# # #     if st.session_state.df is None:
# # #         st.warning("Please upload data first in the 'Upload Data' tab.")
# # #     else:
# # #         df = st.session_state.df_clean.copy()

# # #         # Duplicate detection and removal
# # #         duplicates_count = df.duplicated().sum()
# # #         st.subheader(f"Duplicate Rows: {duplicates_count}")
# # #         if duplicates_count > 0:
# # #             if st.button("Remove Duplicates", key="remove_duplicates"):
# # #                 df.drop_duplicates(inplace=True)
# # #                 st.success(f"Removed {duplicates_count} duplicate rows.")
# # #                 st.session_state.df_clean = df

# # #         # Missing and empty string values summary
# # #         st.subheader("Missing and Empty Values Summary")
# # #         missing = df.isnull().sum()
# # #         empty_str = (df == '').sum()
# # #         missing_df = pd.DataFrame({"Missing Values": missing, "Empty String Values": empty_str})
# # #         missing_df = missing_df[missing_df.sum(axis=1) > 0]
# # #         if missing_df.empty:
# # #             st.info("No missing or empty string values detected.")
# # #         else:
# # #             st.dataframe(missing_df)

# # #             st.subheader("Fill Missing Values")
# # #             fill_option = st.selectbox(
# # #                 "Choose method to handle missing values:",
# # #                 [
# # #                     "None",
# # #                     "Drop rows with missing values",
# # #                     "Fill with Mean (numeric columns only)",
# # #                     "Fill with Median (numeric columns only)",
# # #                     "Fill with Mode (all columns)",
# # #                     "Fill with Custom Values"
# # #                 ]
# # #             )

# # #             custom_values = {}
# # #             if fill_option == "Fill with Custom Values":
# # #                 for col in missing_df.index:
# # #                     val = st.text_input(f"Custom fill value for column '{col}': (leave blank to skip)", key=f"custom_{col}")
# # #                     custom_values[col] = val

# # #             if st.button("Apply missing value handling", key="apply_missing_handling"):
# # #                 if fill_option == "None":
# # #                     st.info("No changes made to missing values.")
# # #                 elif fill_option == "Drop rows with missing values":
# # #                     before_rows = df.shape[0]
# # #                     df.dropna(inplace=True)
# # #                     after_rows = df.shape[0]
# # #                     st.success(f"Dropped {before_rows - after_rows} rows containing missing values.")
# # #                 elif fill_option == "Fill with Mean (numeric columns only)":
# # #                     for col in df.select_dtypes(include=np.number).columns:
# # #                         if col in missing_df.index:
# # #                             df[col].fillna(df[col].mean(), inplace=True)
# # #                     st.success("Filled missing numeric values with mean.")
# # #                 elif fill_option == "Fill with Median (numeric columns only)":
# # #                     for col in df.select_dtypes(include=np.number).columns:
# # #                         if col in missing_df.index:
# # #                             df[col].fillna(df[col].median(), inplace=True)
# # #                     st.success("Filled missing numeric values with median.")
# # #                 elif fill_option == "Fill with Mode (all columns)":
# # #                     for col in missing_df.index:
# # #                         mode_val = df[col].mode()
# # #                         if not mode_val.empty:
# # #                             df[col].fillna(mode_val[0], inplace=True)
# # #                     st.success("Filled missing values with mode.")
# # #                 elif fill_option == "Fill with Custom Values":
# # #                     for col, val in custom_values.items():
# # #                         if val != "":
# # #                             try:
# # #                                 dtype = df[col].dtype
# # #                                 if pd.api.types.is_numeric_dtype(dtype):
# # #                                     val_conv = float(val)
# # #                                 else:
# # #                                     val_conv = val
# # #                                 df[col].fillna(val_conv, inplace=True)
# # #                             except Exception:
# # #                                 st.warning(f"Invalid input for column '{col}'. Skipping.")
# # #                     st.success("Applied custom missing value fills.")
# # #                 st.session_state.df_clean = df

# # #         # Data type conversion suggestions
# # #         st.subheader("Suggested Data Type Conversions")
# # #         type_suggestions = convert_column_types(df)
# # #         if type_suggestions:
# # #             for col, suggested_type in type_suggestions.items():
# # #                 if st.checkbox(f"Convert column '{col}' to {suggested_type}?"):
# # #                     try:
# # #                         if suggested_type == "datetime":
# # #                             df[col] = pd.to_datetime(df[col], errors='coerce')
# # #                         elif suggested_type == "categorical":
# # #                             df[col] = df[col].astype('category')
# # #                         st.success(f"Column '{col}' converted to {suggested_type}.")
# # #                     except Exception as e:
# # #                         st.error(f"Failed to convert column '{col}': {e}")
# # #             st.session_state.df_clean = df
# # #         else:
# # #             st.info("No clear conversions suggested.")

# # #         # Outlier detection and handling
# # #         st.subheader("Outlier Detection and Handling")
# # #         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# # #         selected_outlier_cols = st.multiselect("Select numeric columns to handle outliers", numeric_cols)

# # #         if selected_outlier_cols:
# # #             outlier_method = st.radio("Choose outlier handling method:", ['Cap (Winsorize)', 'Remove Rows'])

# # #             if st.button("Apply Outlier Handling", key="apply_outliers"):
# # #                 df = handle_outliers_iqr(df, selected_outlier_cols, method='cap' if outlier_method == 'Cap (Winsorize)' else 'remove')
# # #                 st.success(f"Outlier handling applied with method: {outlier_method}.")
# # #                 st.session_state.df_clean = df

# # #         # Encoding categorical variables
# # #         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# # #         if categorical_cols:
# # #             st.subheader("Categorical Variable Encoding")
# # #             encode_method = st.selectbox("Select encoding method:", ["None", "One-hot Encoding", "Label Encoding"])
# # #             encode_cols = st.multiselect("Select categorical columns to encode", categorical_cols)

# # #             if encode_method != "None" and encode_cols:
# # #                 if st.button("Apply Encoding", key="apply_encoding"):
# # #                     if encode_method == "One-hot Encoding":
# # #                         df = pd.get_dummies(df, columns=encode_cols)
# # #                         st.success(f"One-hot encoded columns: {encode_cols}")
# # #                     elif encode_method == "Label Encoding":
# # #                         for col in encode_cols:
# # #                             le = LabelEncoder()
# # #                             df[col] = le.fit_transform(df[col].astype(str))
# # #                         st.success(f"Label encoded columns: {encode_cols}")
# # #                     st.session_state.df_clean = df

# # #         # Scaling numeric columns
# # #         if numeric_cols:
# # #             st.subheader("Scale Numeric Columns")
# # #             scale_method = st.selectbox("Choose scaling method:", ["None", "StandardScaler", "MinMaxScaler"])
# # #             scale_cols = st.multiselect("Select numeric columns to scale", numeric_cols)

# # #             if scale_method != "None" and scale_cols:
# # #                 if st.button("Apply Scaling", key="apply_scaling"):
# # #                     scaler = StandardScaler() if scale_method == 'StandardScaler' else MinMaxScaler()
# # #                     df_scaled = df.copy()
# # #                     df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
# # #                     st.session_state.df_clean = df_scaled
# # #                     st.success(f"Applied {scale_method} on columns: {scale_cols}")

# # #         st.subheader("Preview of Preprocessed Data")
# # #         st.dataframe(st.session_state.df_clean.head())

# # #         # Download cleaned data
# # #         cleaned_csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
# # #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv_preprocess")

# # # with tab_visualize:
# # #     st.header("3️⃣ Data Visualization")
# # #     if st.session_state.df_clean is None:
# # #         st.warning("Please upload and preprocess data first.")
# # #     else:
# # #         df = st.session_state.df_clean.copy()
# # #         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# # #         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# # #         st.subheader("Auto Visualizations")

# # #         col1, col2 = st.columns(2)

# # #         with col1:
# # #             if numeric_cols:
# # #                 st.markdown("### Numeric Data Distributions")
# # #                 for col_name in numeric_cols:
# # #                     fig_hist = px.histogram(df, x=col_name, nbins=30, title=f"Histogram of {col_name}", marginal="rug", hover_data=df.columns)
# # #                     st.plotly_chart(fig_hist, use_container_width=True)
# # #                     fig_box = px.box(df, y=col_name, points="outliers", title=f"Boxplot of {col_name}")
# # #                     st.plotly_chart(fig_box, use_container_width=True)

# # #         with col2:
# # #             if categorical_cols:
# # #                 st.markdown("### Categorical Data Counts")
# # #                 for col_name in categorical_cols:
# # #                     counts = df[col_name].value_counts().reset_index()
# # #                     counts.columns = [col_name, 'Count']
# # #                     fig_bar = px.bar(counts, x=col_name, y='Count', title=f"Bar chart of {col_name}")
# # #                     st.plotly_chart(fig_bar, use_container_width=True)

# # #         if len(numeric_cols) > 1:
# # #             st.subheader("Multivariate Numeric Visualizations")
# # #             fig_scatter = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix - Numeric Columns")
# # #             st.plotly_chart(fig_scatter, use_container_width=True)
# # #             corr = df[numeric_cols].corr()
# # #             fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
# # #             st.plotly_chart(fig_heatmap, use_container_width=True)

# # #         st.subheader("Custom Visualization")
# # #         sel_col = st.selectbox("Select column to visualize", df.columns)

# # #         if sel_col in numeric_cols:
# # #             sel_chart = st.selectbox("Select chart type", ["Histogram", "Boxplot", "Line Chart", "Scatter Plot (select another column)"])
# # #         elif sel_col in categorical_cols:
# # #             sel_chart = st.selectbox("Select chart type", ["Bar Chart", "Pie Chart"])
# # #         else:
# # #             sel_chart = None
# # #             st.info("Selected column not supported for custom visualization.")

# # #         if sel_chart and sel_col:
# # #             if sel_chart == "Histogram":
# # #                 fig = px.histogram(df, x=sel_col, nbins=30, title=f"Histogram of {sel_col}")
# # #                 st.plotly_chart(fig, use_container_width=True)
# # #             elif sel_chart == "Boxplot":
# # #                 fig = px.box(df, y=sel_col, title=f"Boxplot of {sel_col}")
# # #                 st.plotly_chart(fig, use_container_width=True)
# # #             elif sel_chart == "Line Chart":
# # #                 fig = px.line(df, y=sel_col, title=f"Line Chart of {sel_col}")
# # #                 st.plotly_chart(fig, use_container_width=True)
# # #             elif sel_chart == "Scatter Plot (select another column)":
# # #                 other_numeric_cols = [c for c in numeric_cols if c != sel_col]
# # #                 if other_numeric_cols:
# # #                     x_col = st.selectbox("Select column for X-axis", other_numeric_cols)
# # #                     fig = px.scatter(df, x=x_col, y=sel_col, title=f"Scatter Plot: {sel_col} vs {x_col}")
# # #                     st.plotly_chart(fig, use_container_width=True)
# # #                 else:
# # #                     st.info("Need at least two numeric columns for scatter plot.")
# # #             elif sel_chart == "Bar Chart":
# # #                 counts = df[sel_col].value_counts().reset_index()
# # #                 counts.columns = [sel_col, "Count"]
# # #                 fig = px.bar(counts, x=sel_col, y="Count", title=f"Bar Chart of {sel_col}")
# # #                 st.plotly_chart(fig, use_container_width=True)
# # #             elif sel_chart == "Pie Chart":
# # #                 fig = px.pie(df, names=sel_col, title=f"Pie Chart of {sel_col}")
# # #                 st.plotly_chart(fig, use_container_width=True)

# # # with tab_insights:
# # #     st.header("4️⃣ Data Insights & Report")
# # #     if st.session_state.df_clean is None:
# # #         st.warning("Please upload and preprocess data first.")
# # #     else:
# # #         df = st.session_state.df_clean.copy()

# # #         st.subheader("Null Value Overview")
# # #         null_sum = df.isnull().sum()
# # #         if null_sum.sum() > 0:
# # #             st.dataframe(null_sum[null_sum > 0].sort_values(ascending=False))
# # #         else:
# # #             st.info("No missing values detected.")

# # #         st.subheader("Outliers Detection using IQR")
# # #         numeric_cols = df.select_dtypes(include=np.number).columns
# # #         outlier_counts = {}
# # #         for col in numeric_cols:
# # #             outlier_idx = detect_outliers_iqr(df[col].dropna())
# # #             outlier_counts[col] = len(outlier_idx)
# # #         outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=["Outlier Count"])
# # #         st.dataframe(outlier_df)

# # #         st.subheader("Skewness of Numeric Features")
# # #         skewness = calc_skewness(df)
# # #         st.dataframe(skewness)

# # #         st.subheader("Cardinality Check")
# # #         cardinalities = cardinality_check(df)
# # #         st.dataframe(cardinalities)

# # #         st.subheader("Strong Correlation (>|0.7|) Pairs")
# # #         corr_issues = correlation_issues(df)
# # #         if corr_issues:
# # #             corr_df = pd.DataFrame(corr_issues, columns=["Feature 1", "Feature 2", "Correlation"])
# # #             st.dataframe(corr_df)
# # #         else:
# # #             st.info("No strong correlations detected.")

# # #         with st.expander("Automated Data Insights Summary"):
# # #             problematic_cols = []
# # #             for col in numeric_cols:
# # #                 skew = df[col].skew()
# # #                 if abs(skew) > 1:
# # #                     problematic_cols.append(f"{col} (skewness={skew:.2f})")
# # #             if problematic_cols:
# # #                 st.write("Highly skewed numeric columns:", ", ".join(problematic_cols))
# # #             else:
# # #                 st.write("No highly skewed columns detected.")

# # #         insights_str = io.StringIO()
# # #         insights_str.write("Explorica Data Insights Summary\n")
# # #         insights_str.write("===============================\n\n")
# # #         insights_str.write("Null Values:\n")
# # #         for col, val in null_sum[null_sum > 0].items():
# # #             insights_str.write(f"{col}: {val}\n")
# # #         insights_str.write("\nOutlier Counts (IQR) per Numeric Column:\n")
# # #         for col, val in outlier_counts.items():
# # #             insights_str.write(f"{col}: {val}\n")
# # #         insights_str.write("\nSkewness of Numeric Columns:\n")
# # #         for col, val in skewness.dropna().items():
# # #             insights_str.write(f"{col}: {val:.4f}\n")
# # #         insights_str.write("\nCardinality per Column:\n")
# # #         for col, val in cardinalities.items():
# # #             insights_str.write(f"{col}: {val}\n")
# # #         if corr_issues:
# # #             insights_str.write("\nStrongly Correlated Pairs (>|0.7|):\n")
# # #             for c1, c2, corr_val in corr_issues:
# # #                 insights_str.write(f"{c1} & {c2}: {corr_val:.4f}\n")
# # #         else:
# # #             insights_str.write("\nNo strong correlations detected.\n")

# # #         st.subheader("Profiling Report")
# # #         if st.session_state.profile_report_html is None:
# # #             with st.spinner("Generating profiling report..."):
# # #                 profile = ProfileReport(df, title="Explorica Profiler", explorative=True)
# # #                 st.session_state.profile_report_html = profile.to_html()
# # #                 st_profile_report(profile)
# # #         else:
# # #             st.markdown("Profiling report loaded from cache.")
# # #             st_profile_report(ProfileReport(df, title="Explorica Profiler", explorative=True))

# # #         cleaned_csv = df.to_csv(index=False).encode('utf-8')
# # #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv_report")
# # #         st.download_button("📥 Download Insights Summary (TXT)", insights_str.getvalue(), "insights_summary.txt", "text/plain", key="download_insights_summary")

# # #         st.subheader("Download Example Visualization: Correlation Heatmap")
# # #         fig, ax = plt.subplots(figsize=(8, 6))
# # #         numeric_df = df.select_dtypes(include=[np.number])
# # #         sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# # #         plt.title("Correlation Heatmap")
# # #         buf = io.BytesIO()
# # #         plt.savefig(buf, format="png")
# # #         buf.seek(0)
# # #         st.download_button("📥 Download Correlation Heatmap", buf, "correlation_heatmap.png", "image/png", key="download_corr_heatmap")
# # #         plt.close(fig)
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import plotly.express as px
# # import io
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# # st.set_page_config(page_title="Explorica - Enhanced Smart EDA", layout="wide")
# # st.title("📊 Explorica - Enhanced Interactive Smart Exploratory Data Analysis Dashboard")

# # # Initialize session state
# # if "df" not in st.session_state:
# #     st.session_state.df = None
# # if "df_clean" not in st.session_state:
# #     st.session_state.df_clean = None
# # if "profile_report_html" not in st.session_state:
# #     st.session_state.profile_report_html = None

# # # Helper functions
# # def detect_outliers_iqr(data):
# #     Q1 = np.percentile(data, 25)
# #     Q3 = np.percentile(data, 75)
# #     IQR = Q3 - Q1
# #     lower_bound = Q1 - 1.5 * IQR
# #     upper_bound = Q3 + 1.5 * IQR
# #     return np.where((data < lower_bound) | (data > upper_bound))[0]

# # def calc_skewness(df):
# #     return df.skew(numeric_only=True)

# # def cardinality_check(df):
# #     return df.nunique()

# # def correlation_issues(df, threshold=0.7):
# #     numeric_df = df.select_dtypes(include=[np.number])
# #     corr = numeric_df.corr().abs()
# #     issues = []
# #     for i in range(len(corr.columns)):
# #         for j in range(i+1, len(corr.columns)):
# #             if corr.iloc[i, j] > threshold:
# #                 issues.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
# #     return issues

# # def convert_column_types(df):
# #     suggestions = {}
# #     for col in df.columns:
# #         if str(df[col].dtype) == 'object':
# #             try:
# #                 pd.to_datetime(df[col], errors='raise')
# #                 suggestions[col] = 'datetime'
# #             except:
# #                 if df[col].nunique() / len(df) < 0.05:
# #                     suggestions[col] = 'categorical'
# #     return suggestions

# # def handle_outliers_iqr(df, cols, method='cap'):
# #     df_out = df.copy()
# #     for col in cols:
# #         data = df_out[col].dropna()
# #         Q1 = np.percentile(data, 25)
# #         Q3 = np.percentile(data, 75)
# #         IQR = Q3 - Q1
# #         lower_bound = Q1 - 1.5 * IQR
# #         upper_bound = Q3 + 1.5 * IQR
# #         if method == 'cap':
# #             df_out.loc[df_out[col] < lower_bound, col] = lower_bound
# #             df_out.loc[df_out[col] > upper_bound, col] = upper_bound
# #         elif method == 'remove':
# #             df_out = df_out.drop(df_out[(df_out[col] < lower_bound) | (df_out[col] > upper_bound)].index)
# #     return df_out

# # def label_encode_columns(df, cols):
# #     df_enc = df.copy()
# #     le = LabelEncoder()
# #     for col in cols:
# #         df_enc[col] = le.fit_transform(df_enc[col].astype(str))
# #     return df_enc

# # # Tabs for navigation
# # tab_upload, tab_preprocess, tab_visualize, tab_insights = st.tabs(
# #     ["Upload Data 🗂️", "Preprocess 🧹", "Visualize 📊", "Insights & Report 📈"]
# # )

# # with tab_upload:
# #     st.header("1️⃣ Upload your CSV file")
# #     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
# #     if uploaded_file is not None:
# #         df = pd.read_csv(uploaded_file)
# #         st.session_state.df = df
# #         st.session_state.df_clean = df.copy()
# #         st.session_state.profile_report_html = None
# #         st.success("✅ Data uploaded successfully!")

# #         with st.expander("Show Data Preview & Summary"):
# #             st.dataframe(df.head())
# #             st.markdown(f"**Data Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# #             dtypes = df.dtypes.astype(str)
# #             col_types = []
# #             for col in df.columns:
# #                 if pd.api.types.is_numeric_dtype(df[col]):
# #                     col_types.append("Numeric")
# #                 elif pd.api.types.is_datetime64_any_dtype(df[col]):
# #                     col_types.append("Datetime")
# #                 elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
# #                     col_types.append("Categorical")
# #                 else:
# #                     col_types.append("Other")
# #             type_df = pd.DataFrame({"Column": df.columns, "Data Type": dtypes.values, "Detected Type": col_types})
# #             st.dataframe(type_df)

# #             st.subheader("Summary Statistics")
# #             st.dataframe(df.describe(include='all').transpose())

# # with tab_preprocess:
# #     st.header("2️⃣ Data Preprocessing")
# #     if st.session_state.df is None:
# #         st.warning("Please upload data first.")
# #     else:
# #         df = st.session_state.df_clean.copy()

# #         duplicates_count = df.duplicated().sum()
# #         st.subheader(f"Duplicate Rows: {duplicates_count}")
# #         if duplicates_count > 0 and st.button("Remove Duplicate Rows", key="remove_duplicates"):
# #             df.drop_duplicates(inplace=True)
# #             st.success(f"Removed {duplicates_count} duplicate rows.")
# #             st.session_state.df_clean = df

# #         st.subheader("Missing and Empty Values Summary")
# #         missing = df.isnull().sum()
# #         empty_str = (df == '').sum()
# #         missing_df = pd.DataFrame({"Missing Values": missing, "Empty String Values": empty_str})
# #         missing_df = missing_df[missing_df.sum(axis=1) > 0]
# #         if missing_df.empty:
# #             st.info("No missing or empty string values detected.")
# #         else:
# #             st.dataframe(missing_df)
# #             st.subheader("Fill Missing Values")
# #             fill_option = st.selectbox(
# #                 "Select missing value handling method:",
# #                 ["None", "Drop rows with missing", "Fill with Mean (numeric only)", "Fill with Median (numeric only)", "Fill with Mode (all)", "Fill with Custom Values"]
# #             )
# #             custom_values = {}
# #             if fill_option == "Fill with Custom Values":
# #                 for col in missing_df.index:
# #                     val = st.text_input(f"Custom fill value for '{col}':", key=f"fill_{col}")
# #                     custom_values[col] = val
# #             if st.button("Apply Missing Value Handling", key="apply_missing"):
# #                 if fill_option == "None":
# #                     st.info("No missing value changes applied.")
# #                 elif fill_option == "Drop rows with missing":
# #                     before_rows = df.shape[0]
# #                     df.dropna(inplace=True)
# #                     after_rows = df.shape[0]
# #                     st.success(f"Dropped {before_rows - after_rows} rows.")
# #                 elif fill_option == "Fill with Mean (numeric only)":
# #                     for col in df.select_dtypes(include=np.number).columns:
# #                         if col in missing_df.index:
# #                             df[col].fillna(df[col].mean(), inplace=True)
# #                     st.success("Filled missing numeric values with mean.")
# #                 elif fill_option == "Fill with Median (numeric only)":
# #                     for col in df.select_dtypes(include=np.number).columns:
# #                         if col in missing_df.index:
# #                             df[col].fillna(df[col].median(), inplace=True)
# #                     st.success("Filled missing numeric values with median.")
# #                 elif fill_option == "Fill with Mode (all)":
# #                     for col in missing_df.index:
# #                         mode_val = df[col].mode()
# #                         if not mode_val.empty:
# #                             df[col].fillna(mode_val[0], inplace=True)
# #                     st.success("Filled missing values with mode.")
# #                 elif fill_option == "Fill with Custom Values":
# #                     for col, val in custom_values.items():
# #                         if val != "":
# #                             try:
# #                                 dtype = df[col].dtype
# #                                 if pd.api.types.is_numeric_dtype(dtype):
# #                                     val_conv = float(val)
# #                                 else:
# #                                     val_conv = val
# #                                 df[col].fillna(val_conv, inplace=True)
# #                             except:
# #                                 st.warning(f"Skipped invalid fill for column '{col}'.")
# #                     st.success("Applied custom fill values.")
# #                 st.session_state.df_clean = df

# #         st.subheader("Suggested Data Type Conversions")
# #         suggestions = convert_column_types(df)
# #         if suggestions:
# #             for col, new_type in suggestions.items():
# #                 if st.checkbox(f"Convert '{col}' to {new_type}?"):
# #                     try:
# #                         if new_type == "datetime":
# #                             df[col] = pd.to_datetime(df[col], errors='coerce')
# #                         elif new_type == "categorical":
# #                             df[col] = df[col].astype('category')
# #                         st.success(f"Converted '{col}' to {new_type}.")
# #                     except Exception as e:
# #                         st.error(f"Failed to convert '{col}': {e}")
# #             st.session_state.df_clean = df
# #         else:
# #             st.info("No type conversion suggestions.")

# #         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# #         st.subheader("Outlier Detection and Treatment")
# #         outlier_cols = st.multiselect("Select columns to treat outliers", numeric_cols)
# #         if outlier_cols:
# #             method = st.radio("Outlier treatment method", ["Cap (Winsorize)", "Remove Rows"])
# #             if st.button("Apply Outlier Treatment", key="apply_outliers"):
# #                 df = handle_outliers_iqr(df, outlier_cols, method='cap' if method == "Cap (Winsorize)" else 'remove')
# #                 st.success(f"Outlier treatment '{method}' applied.")
# #                 st.session_state.df_clean = df

# #         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# #         st.subheader("Categorical Variable Encoding")
# #         enc_method = st.selectbox("Encoding method", ["None", "One-Hot Encoding", "Label Encoding"])
# #         enc_cols = st.multiselect("Select categorical columns to encode", categorical_cols)
# #         if enc_method != "None" and enc_cols:
# #             if st.button("Apply Encoding", key="apply_encoding"):
# #                 if enc_method == "One-Hot Encoding":
# #                     df = pd.get_dummies(df, columns=enc_cols)
# #                     st.success(f"One-hot encoded: {enc_cols}")
# #                 else:
# #                     for col in enc_cols:
# #                         le = LabelEncoder()
# #                         df[col] = le.fit_transform(df[col].astype(str))
# #                     st.success(f"Label encoded: {enc_cols}")
# #                 st.session_state.df_clean = df

# #         st.subheader("Numeric Column Scaling")
# #         scale_cols = st.multiselect("Select numeric columns to scale", numeric_cols)
# #         scale_method = st.selectbox("Choose scaling method", ["None", "StandardScaler", "MinMaxScaler"])
# #         if scale_method != "None" and scale_cols:
# #             if st.button("Apply Scaling", key="apply_scaling"):
# #                 scaler = StandardScaler() if scale_method == "StandardScaler" else MinMaxScaler()
# #                 df_scaled = df.copy()
# #                 df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
# #                 st.session_state.df_clean = df_scaled
# #                 st.success(f"Applied {scale_method} on {scale_cols}")

# #         st.subheader("Data Preview After Preprocessing")
# #         st.dataframe(st.session_state.df_clean.head())

# #         cleaned_csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
# #         st.download_button("📥 Download Cleaned CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv_preproc")

# # with tab_visualize:
# #     st.header("3️⃣ Data Visualization")
# #     if st.session_state.df_clean is None:
# #         st.warning("Please upload and preprocess data first.")
# #     else:
# #         df = st.session_state.df_clean.copy()
# #         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# #         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# #         st.subheader("Auto Visualizations")

# #         col1, col2 = st.columns(2)

# #         with col1:
# #             if numeric_cols:
# #                 st.markdown("### Numeric Distributions")
# #                 for idx, col_name in enumerate(numeric_cols):
# #                     fig_hist = px.histogram(df, x=col_name, nbins=30, title=f"Histogram of {col_name}")
# #                     st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{col_name}")
# #                     fig_box = px.box(df, y=col_name, points="outliers", title=f"Boxplot of {col_name}")
# #                     st.plotly_chart(fig_box, use_container_width=True, key=f"box_{col_name}")

# #         with col2:
# #             if categorical_cols:
# #                 st.markdown("### Categorical Counts")
# #                 for idx, col_name in enumerate(categorical_cols):
# #                     counts_df = df[col_name].value_counts().reset_index()
# #                     counts_df.columns = [col_name, 'count']
# #                     fig_bar = px.bar(counts_df, x=col_name, y='count', title=f"Bar chart of {col_name}")
# #                     st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{col_name}")

# #         if len(numeric_cols) > 1:
# #             st.subheader("Multivariate Numeric Visualizations")
# #             fig_scatter = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix of Numeric Columns")
# #             st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_matrix")
# #             corr = df[numeric_cols].corr()
# #             fig_heat = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
# #             st.plotly_chart(fig_heat, use_container_width=True, key="correlation_heatmap")

# #         st.subheader("Custom Visualization")
# #         sel_col = st.selectbox("Choose column to visualize", df.columns, key="custom_vis_select")
# #         if sel_col in numeric_cols:
# #             sel_chart = st.selectbox("Choose chart type", ["Histogram", "Boxplot", "Line Chart", "Scatter Plot (with X-axis)"], key="custom_vis_chart")
# #         elif sel_col in categorical_cols:
# #             sel_chart = st.selectbox("Choose chart type", ["Bar Chart", "Pie Chart"], key="custom_vis_chart")
# #         else:
# #             sel_chart = None
# #             st.info("Unsupported column type for visualization.")

# #         if sel_chart:
# #             if sel_chart == "Histogram":
# #                 fig = px.histogram(df, x=sel_col, nbins=30, title=f"Histogram of {sel_col}")
# #                 st.plotly_chart(fig, use_container_width=True, key=f"custom_hist_{sel_col}")
# #             elif sel_chart == "Boxplot":
# #                 fig = px.box(df, y=sel_col, title=f"Boxplot of {sel_col}")
# #                 st.plotly_chart(fig, use_container_width=True, key=f"custom_box_{sel_col}")
# #             elif sel_chart == "Line Chart":
# #                 fig = px.line(df, y=sel_col, title=f"Line Chart of {sel_col}")
# #                 st.plotly_chart(fig, use_container_width=True, key=f"custom_line_{sel_col}")
# #             elif sel_chart == "Scatter Plot (with X-axis)":
# #                 other_numeric = [c for c in numeric_cols if c != sel_col]
# #                 if other_numeric:
# #                     x_axis = st.selectbox("Select X-axis column", other_numeric, key="custom_scatter_x")
# #                     fig = px.scatter(df, x=x_axis, y=sel_col, title=f"Scatter Plot of {sel_col} vs {x_axis}")
# #                     st.plotly_chart(fig, use_container_width=True, key=f"custom_scatter_{sel_col}_{x_axis}")
# #                 else:
# #                     st.info("Need at least two numeric columns for scatter plot.")
# #             elif sel_chart == "Bar Chart":
# #                 counts_df = df[sel_col].value_counts().reset_index()
# #                 counts_df.columns = [sel_col, 'count']
# #                 fig = px.bar(counts_df, x=sel_col, y='count', title=f"Bar Chart of {sel_col}")
# #                 st.plotly_chart(fig, use_container_width=True, key=f"custom_bar_{sel_col}")
# #             elif sel_chart == "Pie Chart":
# #                 fig = px.pie(df, names=sel_col, title=f"Pie Chart of {sel_col}")
# #                 st.plotly_chart(fig, use_container_width=True, key=f"custom_pie_{sel_col}")


# # with tab_insights:
# #     st.header("4️⃣ Data Insights & Report")
# #     if st.session_state.df_clean is None:
# #         st.warning("Please upload and preprocess data first.")
# #     else:
# #         df = st.session_state.df_clean.copy()

# #         st.subheader("Null Value Overview")
# #         null_sum = df.isnull().sum()
# #         if null_sum.sum() > 0:
# #             st.dataframe(null_sum[null_sum > 0].sort_values(ascending=False))
# #         else:
# #             st.info("No missing values detected.")

# #         st.subheader("Outliers Detection using IQR")
# #         numeric_cols = df.select_dtypes(include=np.number).columns
# #         outlier_counts = {}
# #         for col in numeric_cols:
# #             outlier_idx = detect_outliers_iqr(df[col].dropna())
# #             outlier_counts[col] = len(outlier_idx)
# #         outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=["Outlier Count"])
# #         st.dataframe(outlier_df)

# #         st.subheader("Skewness of Numeric Features")
# #         skewness = calc_skewness(df)
# #         st.dataframe(skewness)

# #         st.subheader("Cardinality Check")
# #         cardinalities = cardinality_check(df)
# #         st.dataframe(cardinalities)

# #         st.subheader("Strong Correlation (>|0.7|) Pairs")
# #         corr_issues = correlation_issues(df)
# #         if corr_issues:
# #             corr_df = pd.DataFrame(corr_issues, columns=["Feature 1", "Feature 2", "Correlation"])
# #             st.dataframe(corr_df)
# #         else:
# #             st.info("No strong correlations detected.")

# #         # Generate insights summary text for download
# #         insights_str = io.StringIO()
# #         insights_str.write("Explorica Data Insights Summary\n")
# #         insights_str.write("===============================\n\n")
# #         insights_str.write("Null Values:\n")
# #         for col, val in null_sum[null_sum > 0].items():
# #             insights_str.write(f"{col}: {val}\n")
# #         insights_str.write("\nOutlier Counts (IQR) per Numeric Column:\n")
# #         for col, val in outlier_counts.items():
# #             insights_str.write(f"{col}: {val}\n")
# #         insights_str.write("\nSkewness of Numeric Columns:\n")
# #         for col, val in skewness.dropna().items():
# #             insights_str.write(f"{col}: {val:.4f}\n")
# #         insights_str.write("\nCardinality per Column:\n")
# #         for col, val in cardinalities.items():
# #             insights_str.write(f"{col}: {val}\n")
# #         if corr_issues:
# #             insights_str.write("\nStrongly Correlated Pairs (>|0.7|):\n")
# #             for c1, c2, corr_val in corr_issues:
# #                 insights_str.write(f"{c1} & {c2}: {corr_val:.4f}\n")
# #         else:
# #             insights_str.write("\nNo strong correlations detected.\n")

# #         # Display profiling report with caching
# #         st.subheader("Profiling Report")
# #         if st.session_state.profile_report_html is None:
# #             with st.spinner("Generating profiling report..."):
# #                 profile = ProfileReport(df, title="Explorica Profiler", explorative=True)
# #                 st.session_state.profile_report_html = profile.to_html()
# #                 st_profile_report(profile)
# #         else:
# #             st.markdown("Profiling report loaded from cache.")
# #             st_profile_report(ProfileReport(df, title="Explorica Profiler", explorative=True))

# #         # Download cleaned CSV
# #         cleaned_csv = df.to_csv(index=False).encode('utf-8')
# #         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv_report")

# #         # Download insights summary text
# #         st.download_button("📥 Download Insights Summary (TXT)", insights_str.getvalue(), "insights_summary.txt", "text/plain", key="download_insights_summary")

# #         # Download example visualization: Correlation Heatmap as PNG
# #         st.subheader("Download Example Visualization: Correlation Heatmap")

# #         fig, ax = plt.subplots(figsize=(8, 6))
# #         # sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# #         numeric_df = df.select_dtypes(include=[np.number])
# #         sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)

# #         plt.title("Correlation Heatmap")
# #         buf = io.BytesIO()
# #         plt.savefig(buf, format="png")
# #         buf.seek(0)
# #         st.download_button("📥 Download Correlation Heatmap", buf, "correlation_heatmap.png", "image/png", key="download_corr_heatmap")
# #         plt.close(fig)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import io
# import seaborn as sns
# import matplotlib.pyplot as plt
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# st.set_page_config(page_title="Explorica - Enhanced Smart EDA", layout="wide")
# st.title("📊 Explorica - Enhanced Interactive Smart Exploratory Data Analysis Dashboard")

# # Initialize session state
# if "df" not in st.session_state:
#     st.session_state.df = None
# if "df_clean" not in st.session_state:
#     st.session_state.df_clean = None
# if "profile_report_html" not in st.session_state:
#     st.session_state.profile_report_html = None

# # Helper functions
# def detect_outliers_iqr(data):
#     Q1 = np.percentile(data, 25)
#     Q3 = np.percentile(data, 75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return np.where((data < lower_bound) | (data > upper_bound))[0]

# def calc_skewness(df):
#     return df.skew(numeric_only=True)

# def cardinality_check(df):
#     return df.nunique()

# def correlation_issues(df, threshold=0.7):
#     numeric_df = df.select_dtypes(include=[np.number])
#     corr = numeric_df.corr().abs()
#     issues = []
#     for i in range(len(corr.columns)):
#         for j in range(i+1, len(corr.columns)):
#             if corr.iloc[i, j] > threshold:
#                 issues.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
#     return issues

# def convert_column_types(df):
#     suggestions = {}
#     for col in df.columns:
#         if str(df[col].dtype) == 'object':
#             try:
#                 pd.to_datetime(df[col], errors='raise')
#                 suggestions[col] = 'datetime'
#             except:
#                 if df[col].nunique() / len(df) < 0.05:
#                     suggestions[col] = 'categorical'
#     return suggestions

# def handle_outliers_iqr(df, cols, method='cap'):
#     df_out = df.copy()
#     for col in cols:
#         data = df_out[col].dropna()
#         Q1 = np.percentile(data, 25)
#         Q3 = np.percentile(data, 75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         if method == 'cap':
#             df_out.loc[df_out[col] < lower_bound, col] = lower_bound
#             df_out.loc[df_out[col] > upper_bound, col] = upper_bound
#         elif method == 'remove':
#             df_out = df_out.drop(df_out[(df_out[col] < lower_bound) | (df_out[col] > upper_bound)].index)
#     return df_out

# def label_encode_columns(df, cols):
#     df_enc = df.copy()
#     le = LabelEncoder()
#     for col in cols:
#         df_enc[col] = le.fit_transform(df_enc[col].astype(str))
#     return df_enc

# # Tabs for navigation
# tab_upload, tab_preprocess, tab_visualize, tab_insights = st.tabs(
#     ["Upload Data 🗂️", "Preprocess 🧹", "Visualize 📊", "Insights & Report 📈"]
# )

# with tab_upload:
#     st.header("1️⃣ Upload your CSV file")
#     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.session_state.df = df
#         st.session_state.df_clean = df.copy()
#         st.session_state.profile_report_html = None
#         st.success("✅ Data uploaded successfully!")

#         with st.expander("Show Data Preview & Summary"):
#             st.dataframe(df.head())
#             st.markdown(f"**Data Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

#             dtypes = df.dtypes.astype(str)
#             col_types = []
#             for col in df.columns:
#                 if pd.api.types.is_numeric_dtype(df[col]):
#                     col_types.append("Numeric")
#                 elif pd.api.types.is_datetime64_any_dtype(df[col]):
#                     col_types.append("Datetime")
#                 elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
#                     col_types.append("Categorical")
#                 else:
#                     col_types.append("Other")
#             type_df = pd.DataFrame({"Column": df.columns, "Data Type": dtypes.values, "Detected Type": col_types})
#             st.dataframe(type_df)

#             st.subheader("Summary Statistics")
#             st.dataframe(df.describe(include='all').transpose())

# with tab_preprocess:
#     st.header("2️⃣ Data Preprocessing")
#     if st.session_state.df is None:
#         st.warning("Please upload data first.")
#     else:
#         df = st.session_state.df_clean.copy()

#         duplicates_count = df.duplicated().sum()
#         st.subheader(f"Duplicate Rows: {duplicates_count}")
#         if duplicates_count > 0 and st.button("Remove Duplicate Rows", key="remove_duplicates"):
#             df.drop_duplicates(inplace=True)
#             st.success(f"Removed {duplicates_count} duplicate rows.")
#             st.session_state.df_clean = df

#         st.subheader("Missing and Empty Values Summary")
#         missing = df.isnull().sum()
#         empty_str = (df == '').sum()
#         missing_df = pd.DataFrame({"Missing Values": missing, "Empty String Values": empty_str})
#         missing_df = missing_df[missing_df.sum(axis=1) > 0]
#         if missing_df.empty:
#             st.info("No missing or empty string values detected.")
#         else:
#             st.dataframe(missing_df)
#             st.subheader("Fill Missing Values")
#             fill_option = st.selectbox(
#                 "Select missing value handling method:",
#                 ["None", "Drop rows with missing", "Fill with Mean (numeric only)", "Fill with Median (numeric only)", "Fill with Mode (all)", "Fill with Custom Values"]
#             )
#             custom_values = {}
#             if fill_option == "Fill with Custom Values":
#                 for col in missing_df.index:
#                     val = st.text_input(f"Custom fill value for '{col}':", key=f"fill_{col}")
#                     custom_values[col] = val
#             if st.button("Apply Missing Value Handling", key="apply_missing"):
#                 if fill_option == "None":
#                     st.info("No missing value changes applied.")
#                 elif fill_option == "Drop rows with missing":
#                     before_rows = df.shape[0]
#                     df.dropna(inplace=True)
#                     after_rows = df.shape[0]
#                     st.success(f"Dropped {before_rows - after_rows} rows.")
#                 elif fill_option == "Fill with Mean (numeric only)":
#                     for col in df.select_dtypes(include=np.number).columns:
#                         if col in missing_df.index:
#                             df[col].fillna(df[col].mean(), inplace=True)
#                     st.success("Filled missing numeric values with mean.")
#                 elif fill_option == "Fill with Median (numeric only)":
#                     for col in df.select_dtypes(include=np.number).columns:
#                         if col in missing_df.index:
#                             df[col].fillna(df[col].median(), inplace=True)
#                     st.success("Filled missing numeric values with median.")
#                 elif fill_option == "Fill with Mode (all)":
#                     for col in missing_df.index:
#                         mode_val = df[col].mode()
#                         if not mode_val.empty:
#                             df[col].fillna(mode_val[0], inplace=True)
#                     st.success("Filled missing values with mode.")
#                 elif fill_option == "Fill with Custom Values":
#                     for col, val in custom_values.items():
#                         if val != "":
#                             try:
#                                 dtype = df[col].dtype
#                                 if pd.api.types.is_numeric_dtype(dtype):
#                                     val_conv = float(val)
#                                 else:
#                                     val_conv = val
#                                 df[col].fillna(val_conv, inplace=True)
#                             except:
#                                 st.warning(f"Skipped invalid fill for column '{col}'.")
#                     st.success("Applied custom fill values.")
#                 st.session_state.df_clean = df

#         st.subheader("Suggested Data Type Conversions")
#         suggestions = convert_column_types(df)
#         if suggestions:
#             for col, new_type in suggestions.items():
#                 if st.checkbox(f"Convert '{col}' to {new_type}?"):
#                     try:
#                         if new_type == "datetime":
#                             df[col] = pd.to_datetime(df[col], errors='coerce')
#                         elif new_type == "categorical":
#                             df[col] = df[col].astype('category')
#                         st.success(f"Converted '{col}' to {new_type}.")
#                     except Exception as e:
#                         st.error(f"Failed to convert '{col}': {e}")
#             st.session_state.df_clean = df
#         else:
#             st.info("No type conversion suggestions.")

#         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#         st.subheader("Outlier Detection and Treatment")
#         outlier_cols = st.multiselect("Select columns to treat outliers", numeric_cols)
#         if outlier_cols:
#             method = st.radio("Outlier treatment method", ["Cap (Winsorize)", "Remove Rows"])
#             if st.button("Apply Outlier Treatment", key="apply_outliers"):
#                 df = handle_outliers_iqr(df, outlier_cols, method='cap' if method == "Cap (Winsorize)" else 'remove')
#                 st.success(f"Outlier treatment '{method}' applied.")
#                 st.session_state.df_clean = df

#         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#         st.subheader("Categorical Variable Encoding")
#         enc_method = st.selectbox("Encoding method", ["None", "One-Hot Encoding", "Label Encoding"])
#         enc_cols = st.multiselect("Select categorical columns to encode", categorical_cols)
#         if enc_method != "None" and enc_cols:
#             if st.button("Apply Encoding", key="apply_encoding"):
#                 if enc_method == "One-Hot Encoding":
#                     df = pd.get_dummies(df, columns=enc_cols)
#                     st.success(f"One-hot encoded: {enc_cols}")
#                 else:
#                     for col in enc_cols:
#                         le = LabelEncoder()
#                         df[col] = le.fit_transform(df[col].astype(str))
#                     st.success(f"Label encoded: {enc_cols}")
#                 st.session_state.df_clean = df

#         st.subheader("Numeric Column Scaling")
#         scale_cols = st.multiselect("Select numeric columns to scale", numeric_cols)
#         scale_method = st.selectbox("Choose scaling method", ["None", "StandardScaler", "MinMaxScaler"])
#         if scale_method != "None" and scale_cols:
#             if st.button("Apply Scaling", key="apply_scaling"):
#                 scaler = StandardScaler() if scale_method == "StandardScaler" else MinMaxScaler()
#                 df_scaled = df.copy()
#                 df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
#                 st.session_state.df_clean = df_scaled
#                 st.success(f"Applied {scale_method} on {scale_cols}")

#         st.subheader("Data Preview After Preprocessing")
#         st.dataframe(st.session_state.df_clean.head())

#         cleaned_csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             "📥 Download Cleaned CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_csv_preproc"
#         )

# with tab_visualize:
#     st.header("3️⃣ Data Visualization")
#     if st.session_state.df_clean is None:
#         st.warning("Please upload and preprocess data first.")
#     else:
#         df = st.session_state.df_clean.copy()
#         numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

#         st.subheader("Auto Visualizations")

#         col1, col2 = st.columns(2)

#         with col1:
#             if numeric_cols:
#                 st.markdown("### Numeric Distributions")
#                 for col_name in numeric_cols:
#                     fig_hist = px.histogram(df, x=col_name, nbins=30, title=f"Histogram of {col_name}")
#                     st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{col_name}")
#                     fig_box = px.box(df, y=col_name, points="outliers", title=f"Boxplot of {col_name}")
#                     st.plotly_chart(fig_box, use_container_width=True, key=f"box_{col_name}")

#         with col2:
#             if categorical_cols:
#                 st.markdown("### Categorical Counts")
#                 for col_name in categorical_cols:
#                     counts_df = df[col_name].value_counts().reset_index()
#                     counts_df.columns = [col_name, 'count']
#                     fig_bar = px.bar(counts_df, x=col_name, y='count', title=f"Bar chart of {col_name}")
#                     st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{col_name}")

#         if len(numeric_cols) > 1:
#             st.subheader("Multivariate Numeric Visualizations")
#             fig_scatter = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix of Numeric Columns")
#             st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_matrix")
#             corr = df[numeric_cols].corr()
#             fig_heat = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
#             st.plotly_chart(fig_heat, use_container_width=True, key="correlation_heatmap")

#         st.subheader("Custom Visualization")
#         sel_col = st.selectbox("Choose column to visualize", df.columns, key="custom_vis_select")
#         if sel_col in numeric_cols:
#             sel_chart = st.selectbox(
#                 "Choose chart type",
#                 ["Histogram", "Boxplot", "Line Chart", "Scatter Plot (with X-axis)"],
#                 key="custom_vis_chart",
#             )
#         elif sel_col in categorical_cols:
#             sel_chart = st.selectbox("Choose chart type", ["Bar Chart", "Pie Chart"], key="custom_vis_chart")
#         else:
#             sel_chart = None
#             st.info("Unsupported column type for visualization.")

#         if sel_chart:
#             if sel_chart == "Histogram":
#                 fig = px.histogram(df, x=sel_col, nbins=30, title=f"Histogram of {sel_col}")
#                 st.plotly_chart(fig, use_container_width=True, key=f"custom_hist_{sel_col}")
#             elif sel_chart == "Boxplot":
#                 fig = px.box(df, y=sel_col, title=f"Boxplot of {sel_col}")
#                 st.plotly_chart(fig, use_container_width=True, key=f"custom_box_{sel_col}")
#             elif sel_chart == "Line Chart":
#                 fig = px.line(df, y=sel_col, title=f"Line Chart of {sel_col}")
#                 st.plotly_chart(fig, use_container_width=True, key=f"custom_line_{sel_col}")
#             elif sel_chart == "Scatter Plot (with X-axis)":
#                 other_numeric = [c for c in numeric_cols if c != sel_col]
#                 if other_numeric:
#                     x_axis = st.selectbox("Select X-axis column", other_numeric, key="custom_scatter_x")
#                     fig = px.scatter(df, x=x_axis, y=sel_col, title=f"Scatter Plot of {sel_col} vs {x_axis}")
#                     st.plotly_chart(fig, use_container_width=True, key=f"custom_scatter_{sel_col}_{x_axis}")
#                 else:
#                     st.info("Need at least two numeric columns for scatter plot.")
#             elif sel_chart == "Bar Chart":
#                 counts_df = df[sel_col].value_counts().reset_index()
#                 counts_df.columns = [sel_col, "count"]
#                 fig = px.bar(counts_df, x=sel_col, y="count", title=f"Bar Chart of {sel_col}")
#                 st.plotly_chart(fig, use_container_width=True, key=f"custom_bar_{sel_col}")
#             elif sel_chart == "Pie Chart":
#                 fig = px.pie(df, names=sel_col, title=f"Pie Chart of {sel_col}")
#                 st.plotly_chart(fig, use_container_width=True, key=f"custom_pie_{sel_col}")

# with tab_insights:
#     st.header("4️⃣ Data Insights & Final Report")
#     if st.session_state.df_clean is None:
#         st.warning("Please upload and preprocess data first.")
#     else:
#         df = st.session_state.df_clean.copy()

#         # Null Value Overview
#         st.subheader("Null Value Overview")
#         null_sum = df.isnull().sum()
#         if null_sum.sum() > 0:
#             st.dataframe(null_sum[null_sum > 0].sort_values(ascending=False))
#         else:
#             st.info("No missing values detected.")

#         # Outliers Detection using IQR
#         st.subheader("Outliers Detection (IQR Method)")
#         numeric_cols = df.select_dtypes(include=np.number).columns
#         outlier_counts = {}
#         for col in numeric_cols:
#             outlier_idx = detect_outliers_iqr(df[col].dropna())
#             outlier_counts[col] = len(outlier_idx)
#         outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=["Outlier Count"])
#         st.dataframe(outlier_df)

#         # Skewness of Numeric Features
#         st.subheader("Skewness of Numeric Features")
#         skewness = calc_skewness(df)
#         st.dataframe(skewness)

#         # Cardinality Check
#         st.subheader("Cardinality (Unique Values) per Column")
#         cardinalities = cardinality_check(df)
#         st.dataframe(cardinalities)

#         # Strong Correlation Pairs
#         st.subheader("Strong Correlation Pairs (Absolute Correlation > 0.7)")
#         corr_issues = correlation_issues(df)
#         if corr_issues:
#             corr_df = pd.DataFrame(corr_issues, columns=["Feature 1", "Feature 2", "Correlation"])
#             st.dataframe(corr_df)
#         else:
#             st.info("No strong correlations detected.")

#         # Automated Insights Summary
#         with st.expander("Automated Data Insights Summary"):
#             highly_skewed = [f"{col} (skewness={skewness[col]:.2f})" for col in numeric_cols if abs(skewness[col]) > 1]
#             if highly_skewed:
#                 st.write("Highly skewed numeric columns detected:", ", ".join(highly_skewed))
#             else:
#                 st.write("No highly skewed columns detected.")
#             high_missing = null_sum[null_sum > 0]
#             if not high_missing.empty:
#                 st.write("Columns with missing values:")
#                 for col, cnt in high_missing.items():
#                     st.write(f"- {col}: {cnt} missing values")
#             else:
#                 st.write("No missing data detected.")

#         # Prepare textual insights summary for download
#         insights_text = io.StringIO()
#         insights_text.write("Explorica Data Insights Summary\n===============================\n\n")
#         insights_text.write("Missing Values:\n")
#         for col, val in null_sum[null_sum > 0].items():
#             insights_text.write(f"{col}: {val}\n")
#         insights_text.write("\nOutlier Counts (IQR):\n")
#         for col, val in outlier_counts.items():
#             insights_text.write(f"{col}: {val}\n")
#         insights_text.write("\nSkewness Values:\n")
#         for col, val in skewness.dropna().items():
#             insights_text.write(f"{col}: {val:.4f}\n")
#         insights_text.write("\nCardinality Per Column:\n")
#         for col, val in cardinalities.items():
#             insights_text.write(f"{col}: {val}\n")
#         if corr_issues:
#             insights_text.write("\nStrong Correlations (|correlation| > 0.7):\n")
#             for f1, f2, corr_v in corr_issues:
#                 insights_text.write(f"{f1} & {f2}: {corr_v:.4f}\n")
#         else:
#             insights_text.write("\nNo strong correlations detected.\n")

#         st.subheader("Profiling Report (Manual)")
#         st.info("This panel provides a manual alternative to automated profiling with interactive visualizations and computed summaries.")

#         cleaned_csv = df.to_csv(index=False).encode('utf-8')
#         st.download_button("📥 Download Cleaned Data CSV", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_clean")

#         st.download_button("📥 Download Insights Summary (TXT)", insights_text.getvalue(), "insights_summary.txt", "text/plain", key="download_insights")

#         st.subheader("Download Sample Visualization: Correlation Heatmap")
#         fig, ax = plt.subplots(figsize=(8, 6))
#         numeric_df = df.select_dtypes(include=[np.number])
#         sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
#         plt.title("Correlation Heatmap")
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         st.download_button("📥 Download Correlation Heatmap", buf, "correlation_heatmap.png", "image/png", key="download_heatmap")
#         plt.close(fig)


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

st.set_page_config(page_title="Explorica - Smart EDA", layout="wide")
st.title("📊 Explorica - Smart Exploratory Data Analysis Dashboard")

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "profile_report" not in st.session_state:
    st.session_state.profile_report = None

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
    corr = numeric_df.corr()
    corr_abs = corr.abs()
    issues = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr_abs.iloc[i, j] > threshold:
                issues.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    return issues

def convert_column_types(df):
    suggestions = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                suggestions[col] = 'datetime'
            except:
                if df[col].nunique() / len(df) < 0.05:
                    suggestions[col] = 'categorical'
    return suggestions

def handle_outliers_iqr(df, cols, method='cap'):
    df_out = df.copy()
    for col in cols:
        data = df_out[col].dropna()
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if method == 'cap':
            df_out.loc[df_out[col] < lower_bound, col] = lower_bound
            df_out.loc[df_out[col] > upper_bound, col] = upper_bound
        elif method == 'remove':
            df_out = df_out.drop(df_out[(df_out[col] < lower_bound) | (df_out[col] > upper_bound)].index)
    return df_out

def label_encode_columns(df, cols):
    df_enc = df.copy()
    le = LabelEncoder()
    for col in cols:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc

# Layout tabs
tab_upload, tab_preprocess, tab_visualize, tab_insights = st.tabs(
    ["Upload Data 🗂️", "Preprocess 🧹", "Visualize 📊", "Insights & Report 📈"]
)

with tab_upload:
    st.header("1️⃣ Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.df_clean = df.copy()
        st.session_state.profile_report = None
        st.success("File uploaded successfully!")

        with st.expander("Data Preview and Summary"):
            st.dataframe(df.head())
            st.write(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
            dtypes = df.dtypes.astype(str)
            col_types = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_types.append('Numeric')
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_types.append('Datetime')
                elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
                    col_types.append('Categorical')
                else:
                    col_types.append('Other')
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Dtype': dtypes.values,
                'Detected Type': col_types
            })
            st.dataframe(dtype_df)
            st.write(df.describe(include='all').transpose())

with tab_preprocess:
    st.header("2️⃣ Preprocessing")
    if st.session_state.df is None:
        st.warning("Upload dataset first.")
    else:
        df = st.session_state.df_clean.copy()

        duplicates = df.duplicated().sum()
        st.write(f"Duplicate rows: {duplicates}")
        if duplicates > 0 and st.button("Remove Duplicates", key="remove_dup"):
            df = df.drop_duplicates()
            st.success(f"Removed {duplicates} duplicate rows.")
            st.session_state.df_clean = df

        # Missing data summary and treatment
        missing_counts = df.isnull().sum()
        empty_counts = (df == '').sum()
        missing_display = pd.DataFrame({'Missing': missing_counts, 'Empty': empty_counts})
        missing_display = missing_display[(missing_display.Missing > 0) | (missing_display.Empty > 0)]
        if missing_display.empty:
            st.info("No missing or empty values detected.")
        else:
            st.dataframe(missing_display)

            fill_method = st.selectbox("Fill missing values method", ["None", "Drop Rows", "Mean", "Median", "Mode", "Custom"])
            custom_fill_values = {}
            if fill_method == 'Custom':
                for column in missing_display.index:
                    val = st.text_input(f"Custom fill for column: {column}", key=f'custom_{column}')
                    custom_fill_values[column] = val

            if st.button("Apply missing value treatment", key='apply_missing_treatment'):
                if fill_method == 'None':
                    st.info("No missing value imputation applied.")
                elif fill_method == 'Drop Rows':
                    before_rows = df.shape[0]
                    df = df.dropna()
                    after_rows = df.shape[0]
                    st.success(f"Dropped {before_rows - after_rows} rows containing missing values.")
                elif fill_method == 'Mean':
                    for col in df.select_dtypes(include=np.number).columns:
                        df[col] = df[col].fillna(df[col].mean())
                    st.success("Filled missing numeric values with mean.")
                elif fill_method == 'Median':
                    for col in df.select_dtypes(include=np.number).columns:
                        df[col] = df[col].fillna(df[col].median())
                    st.success("Filled missing numeric values with median.")
                elif fill_method == 'Mode':
                    for col in df.columns:
                        mode = df[col].mode()
                        if not mode.empty:
                            df[col] = df[col].fillna(mode[0])
                    st.success("Filled missing values with mode.")
                elif fill_method == 'Custom':
                    for col, val in custom_fill_values.items():
                        if val:
                            try:
                                dtype = df[col].dtype
                                if pd.api.types.is_numeric_dtype(dtype):
                                    fill_val = float(val)
                                else:
                                    fill_val = val
                                df[col] = df[col].fillna(fill_val)
                            except:
                                st.warning(f"Skipping invalid custom fill value for column {col}.")
                    st.success("Applied custom fill values.")
                st.session_state.df_clean = df

        # Suggested type conversions
        suggestions = convert_column_types(df)
        if suggestions:
            st.write("Suggested Type Conversions:")
            for col, conv_type in suggestions.items():
                if st.checkbox(f"Convert '{col}' to {conv_type}", key=f"conv_{col}"):
                    try:
                        if conv_type == 'datetime':
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif conv_type == 'categorical':
                            df[col] = df[col].astype('category')
                        st.success(f"Converted column '{col}' to {conv_type}.")
                    except Exception as e:
                        st.error(f"Failed to convert '{col}': {e}")
                    st.session_state.df_clean = df

        # Outlier handling
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_outliers = st.multiselect("Select columns for outlier treatment", numeric_cols)
        if selected_outliers:
            outlier_method = st.radio("Outlier treatment method", ['Cap (Winsorize)', 'Remove rows'])
            if st.button("Apply outlier treatment", key='apply_outlier_treatment'):
                df = handle_outliers_iqr(df, selected_outliers, method='cap' if outlier_method == 'Cap (Winsorize)' else 'remove')
                st.success(f"Applied outlier treatment: {outlier_method}")
                st.session_state.df_clean = df

        # Encoding categorical vars
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            encoding_method = st.selectbox("Select encoding method", ['None', 'One-Hot', 'Label Encoding'])
            selected_encode_cols = st.multiselect("Select categorical columns to encode", cat_cols)
            if encoding_method != 'None' and selected_encode_cols:
                if st.button("Apply encoding", key='apply_encoding'):
                    if encoding_method == 'One-Hot':
                        df = pd.get_dummies(df, columns=selected_encode_cols)
                    else:
                        le = LabelEncoder()
                        for col in selected_encode_cols:
                            df[col] = le.fit_transform(df[col].astype(str))
                    st.success(f"Applied {encoding_method} encoding on columns: {selected_encode_cols}")
                    st.session_state.df_clean = df

        # Scaling numeric vars
        if numeric_cols:
            scale_method = st.selectbox("Select scaling method", ['None', 'StandardScaler', 'MinMaxScaler'])
            selected_scale_cols = st.multiselect("Select numeric columns to scale", numeric_cols)
            if scale_method != 'None' and selected_scale_cols:
                if st.button("Apply scaling", key='apply_scaling'):
                    scaler = StandardScaler() if scale_method == 'StandardScaler' else MinMaxScaler()
                    df[selected_scale_cols] = scaler.fit_transform(df[selected_scale_cols])
                    st.success(f"Applied {scale_method} on columns: {selected_scale_cols}")
                    st.session_state.df_clean = df

        st.write("Data Preview after Preprocessing:")
        st.dataframe(st.session_state.df_clean.head())

        cleaned_csv = st.session_state.df_clean.to_csv(index=False).encode('utf-8')
        # st.download_button("Download Cleaned Data", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_data")
        st.download_button("Download Cleaned Data", cleaned_csv, "cleaned_data.csv", "text/csv", key="download_cleaned_preproc")

with tab_visualize:
    st.header("3️⃣ Visualization")
    if st.session_state.df_clean is None:
        st.warning("Please upload and preprocess data first.")
    else:
        df = st.session_state.df_clean.copy()

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
                count_df = df[col].value_counts().reset_index()
                count_df.columns = [col, 'count']
                bar_fig = px.bar(count_df, x=col, y='count', title=f"Bar chart of {col}")
                st.plotly_chart(bar_fig, use_container_width=True, key=f"bar_{col}_{i}")

        if len(num_cols) > 1:
            st.subheader("Pairwise numeric visualization")
            pair_fig = px.scatter_matrix(df, dimensions=num_cols)
            st.plotly_chart(pair_fig, use_container_width=True, key="pairwise_numeric")

            corr_matrix = df[num_cols].corr()
            corr_fig = px.imshow(corr_matrix, text_auto=True)
            st.plotly_chart(corr_fig, use_container_width=True, key="correlation_heatmap")

        st.subheader("Custom Visualization")
        chosen_col = st.selectbox('Select column for custom visualization', df.columns, key='custom_vis_col')

        if chosen_col in num_cols:
            chart_type = st.selectbox('Select chart type', ['Histogram', 'Box plot', 'Line chart', 'Scatter plot with another numeric column'], key='custom_numeric_chart')
        elif chosen_col in cat_cols:
            chart_type = st.selectbox('Select chart type', ['Bar chart', 'Pie chart'], key='custom_cat_chart')
        else:
            chart_type = None
            st.info('Unsupported column type for visualization.')

        if chart_type:
            if chart_type == 'Histogram':
                fig = px.histogram(df, x=chosen_col)
                st.plotly_chart(fig, use_container_width=True, key='custom_hist')
            elif chart_type == 'Box plot':
                fig = px.box(df, y=chosen_col)
                st.plotly_chart(fig, use_container_width=True, key='custom_box')
            elif chart_type == 'Line chart':
                fig = px.line(df, y=chosen_col)
                st.plotly_chart(fig, use_container_width=True, key='custom_line')
            elif chart_type == 'Scatter plot with another numeric column':
                other_cols = [c for c in num_cols if c != chosen_col]
                if other_cols:
                    x_col = st.selectbox('Select X-axis column', other_cols, key='scatter_x_col')
                    fig = px.scatter(df, x=x_col, y=chosen_col)
                    st.plotly_chart(fig, use_container_width=True, key='custom_scatter')
                else:
                    st.info("No other numeric columns available.")
            elif chart_type == 'Bar chart':
                count_df = df[chosen_col].value_counts().reset_index()
                count_df.columns = [chosen_col, 'count']
                fig = px.bar(count_df, x=chosen_col, y='count')
                st.plotly_chart(fig, use_container_width=True, key='custom_bar')
            elif chart_type == 'Pie chart':
                fig = px.pie(df, names=chosen_col)
                st.plotly_chart(fig, use_container_width=True, key='custom_pie')

with tab_insights:
    st.header("4️⃣ Insights and Final Report")
    if st.session_state.df_clean is None:
        st.warning("Please upload and preprocess data first.")
    else:
        df = st.session_state.df_clean.copy()
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

        with st.expander("Automated Insights Summary"):
            high_skew_cols = [f"{col} (skew={skewness[col]:.2f})" for col in skewness.index if abs(skewness[col]) > 1]
            if high_skew_cols:
                st.write("Highly skewed columns:", ", ".join(high_skew_cols))
            else:
                st.write("No strong skewness detected.")
            if not missing_vals.empty:
                st.write("Columns with missing data:")
                for col, val in missing_vals.items():
                    st.write(f"- {col}: {val}")
            else:
                st.write("No missing data.")

        insights_text = io.StringIO()
        insights_text.write("Explorica Data Summary\n=====================\n\n")
        insights_text.write("Missing Values:\n")
        for col, val in missing_vals.items():
            insights_text.write(f"{col}: {val}\n")
        insights_text.write("\nOutlier Counts:\n")
        for col, val in outlier_counts.items():
            insights_text.write(f"{col}: {val}\n")
        insights_text.write("\nSkewness:\n")
        for col, val in skewness.dropna().items():
            insights_text.write(f"{col}: {val:.2f}\n")
        insights_text.write("\nCardinality:\n")
        for col, val in cardinality.items():
            insights_text.write(f"{col}: {val}\n")
        if corr_pairs:
            insights_text.write("\nHighly Correlated Pairs:\n")
            for f1, f2, corr_val in corr_pairs:
                insights_text.write(f"{f1} & {f2}: {corr_val:.2f}\n")
        else:
            insights_text.write("\nNo highly correlated pairs found.\n")

        st.subheader("Profiling Report")
        if st.session_state.profile_report is None:
            with st.spinner("Generating profiling report..."):
                profile = ProfileReport(df, title="Explorica Profiling Report", explorative=True)
                st.session_state.profile_report = profile
                st_profile_report(profile)
        else:
            st_profile_report(st.session_state.profile_report)

        cleaned_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned Data", cleaned_csv, "cleaned_data.csv", "text/csv", key='download_cleaned_data')

        st.download_button("Download Insights Summary", insights_text.getvalue(), "insights_summary.txt", "text/plain", key='download_insights')

        st.subheader("Download Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        plt.title("Correlation Heatmap")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        st.download_button("Download Heatmap", buf, "correlation_heatmap.png", "image/png", key='download_heatmap')
