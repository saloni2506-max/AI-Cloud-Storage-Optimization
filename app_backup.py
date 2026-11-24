
import streamlit as st
import pandas as pd
import plotly.express as px
import os

from drive_api import scan_drive_files
from unused_file_detection import analyze_unused_files
from duplicate_detection import detect_duplicate_files
from drive_api import move_file_to_trash
# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Cloud Storage Optimizer",
    page_icon="☁️",
    layout="wide"
)

# ==========================================
# 2. CUSTOM STYLE
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #F5F7FB;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
}

h1, h2, h3 {
    color: #172554;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. FILE TYPE CONVERSION
# ==========================================

mime_names = {
    "application/pdf": "PDF",
    "image/jpeg": "JPEG Image",
    "image/png": "PNG Image",
    "application/vnd.google-apps.document": "Google Docs",
    "application/vnd.google-apps.spreadsheet": "Google Sheets",
    "application/vnd.google-apps.presentation": "Google Slides",
    "application/vnd.google-apps.folder": "Folder",
    "application/vnd.google-apps.shortcut": "Shortcut",
    "application/vnd.google.colaboratory": "Google Colab",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word Document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel Workbook",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PowerPoint",
    "application/x-msi": "MSI File",
    "text/plain": "Text File",
    "video/mp4": "MP4 Video",
    "video/webm": "WebM Video"
}


def readable_file_types(df):

    display_df = df.copy()

    if "file_type" in display_df.columns:
        display_df["file_type"] = (
            display_df["file_type"]
            .fillna("Unknown")
            .map(lambda x: mime_names.get(x, x))
        )

    return display_df


# ==========================================
# 4. LOAD PROJECT DATA
# ==========================================

@st.cache_data
def load_data():

    filenames = {
        "drive": "google_drive_files.csv",
        "unused": "unused_file_analysis.csv",
        "duplicates": "duplicate_files.csv",
        "predictions": "drive_ml_predictions.csv"
    }

    files = {}

    for key, filename in filenames.items():

        if os.path.exists(filename):
            files[key] = pd.read_csv(filename)
        else:
            files[key] = pd.DataFrame()

    return files


data = load_data()

drive = data["drive"]
unused = data["unused"]
duplicates = data["duplicates"]
predictions = data["predictions"]


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def find_column(df, candidates):

    for col in candidates:
        if col in df.columns:
            return col

    return None


def show_dataframe(df):

    if not df.empty:
        st.dataframe(
            readable_file_types(df),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No data available.")


def get_duplicate_savings(df):

    """
    Estimate extra storage occupied by duplicate copies.
    Keeps one copy per MD5 checksum group.
    """

    required = ["md5_checksum", "file_size_mb"]

    if df.empty or not all(col in df.columns for col in required):
        return 0.0

    duplicate_data = df.copy()

    duplicate_data["file_size_mb"] = pd.to_numeric(
        duplicate_data["file_size_mb"],
        errors="coerce"
    )

    duplicate_data = duplicate_data.dropna(
        subset=["md5_checksum", "file_size_mb"]
    )

    duplicate_data = duplicate_data[
        duplicate_data["md5_checksum"].astype(str).str.strip() != ""
    ]

    if duplicate_data.empty:
        return 0.0

    savings = (
        duplicate_data.groupby("md5_checksum")["file_size_mb"]
        .sum()
        -
        duplicate_data.groupby("md5_checksum")["file_size_mb"]
        .max()
    )

    return max(float(savings.sum()), 0.0)


def get_archive_summary(df, recommendation_col):

    if df.empty or not recommendation_col:
        return 0, 0.0

    archive_files = df[
        df[recommendation_col]
        .astype(str)
        .str.strip()
        .str.lower() == "archive"
    ]

    archive_count = len(archive_files)

    if "file_size_mb" in archive_files.columns:

        archive_size = pd.to_numeric(
            archive_files["file_size_mb"],
            errors="coerce"
        ).sum()

    else:
        archive_size = 0.0

    return archive_count, float(archive_size)


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("☁️ AI Cloud Optimizer")
 # ==========================================
# GOOGLE DRIVE LIVE SCAN
# ==========================================

st.sidebar.subheader("Google Drive")


if st.sidebar.button("🔄 Scan My Drive", use_container_width=True):

    try:
        with st.spinner(
            "Scanning Google Drive, analysing files and detecting duplicates..."
        ):

            # Step 1: Scan Google Drive
            scanned_data = scan_drive_files()

            # Step 2: Analyse unused files
            analyzed_data = analyze_unused_files(scanned_data)

            # Step 3: Detect duplicate files
            duplicate_data, duplicate_summary = detect_duplicate_files(
                scanned_data
            )

        # Step 4: Clear cached dashboard data
        st.cache_data.clear()

        # Step 5: Display success message
        st.session_state["drive_scan_message"] = (
            f"Successfully scanned {len(scanned_data)} files, "
            f"analysed {len(analyzed_data)} files and "
            f"detected {len(duplicate_data)} duplicate entries!"
        )

        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Scan failed: {e}")


if st.session_state.get("drive_scan_message"):

    st.sidebar.success(
        st.session_state["drive_scan_message"]
    )

    del st.session_state["drive_scan_message"]

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Google Drive Explorer",
        "Unused File Analysis",
        "Duplicate Detection",
        "AI Predictions"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "AI-Based Unused File Detection and Cloud Storage Optimization"
)


# ==========================================
# 7. DASHBOARD OVERVIEW
# ==========================================

if page == "Dashboard Overview":

    st.title("☁️ AI Cloud Storage Optimizer")

    st.write(
        "Intelligent analysis of Google Drive files "
        "using Machine Learning and data analytics."
    )

    if drive.empty:
        st.error("Google Drive dataset not found.")
        st.stop()

    # Basic metrics

    total_files = len(drive)

    total_size = 0.0

    if "file_size_mb" in drive.columns:
        total_size = pd.to_numeric(
            drive["file_size_mb"],
            errors="coerce"
        ).sum()

    duplicate_groups = 0

    if not duplicates.empty and "md5_checksum" in duplicates.columns:
        duplicate_groups = duplicates[
            "md5_checksum"
        ].dropna().nunique()

    ml_count = len(predictions)

    # Main KPI cards

    st.subheader("Google Drive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Drive Records",
        total_files
    )

    col2.metric(
        "Storage Used",
        f"{total_size:,.2f} MB"
    )

    col3.metric(
        "Duplicate Groups",
        duplicate_groups
    )

    col4.metric(
        "ML Predictions",
        ml_count
    )

    st.divider()

    # File type distribution

    st.subheader("Google Drive File Type Distribution")

    if "file_type" in drive.columns:

        file_summary = (
            drive["file_type"]
            .fillna("Unknown")
            .map(lambda x: mime_names.get(x, x))
            .value_counts()
            .rename_axis("File Type")
            .reset_index(name="Count")
        )

        fig = px.bar(
            file_summary,
            x="File Type",
            y="Count",
            title="Files by Type",
            color="Count",
            template="plotly_white"
        )

        st.plotly_chart(fig, width="stretch")

    # File type and project information

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("File Type Summary")

        if "file_type" in drive.columns:

            file_type_summary = (
                drive["file_type"]
                .fillna("Unknown")
                .map(lambda x: mime_names.get(x, x))
                .value_counts()
                .rename_axis("File Type")
                .reset_index(name="Count")
            )

            st.dataframe(
                file_type_summary,
                width="stretch",
                hide_index=True
            )

    with col2:

        st.subheader("Project Information")

        st.write("**Project:** AI-Based Unused File Detection")
        st.write("**Data source:** Google Drive")
        st.write("**ML algorithm:** Random Forest")

        st.write(
            "**Other algorithms evaluated:** "
            "Decision Tree, Logistic Regression"
        )

    # STORAGE OPTIMIZATION SUMMARY

    st.divider()

    st.header("💾 Storage Optimization Summary")

    recommendation_col = find_column(
        unused,
        [
            "recommendation",
            "Recommendation",
            "suggested_action",
            "status",
            "action",
            "category"
        ]
    )

    archive_count, archive_size = get_archive_summary(
        unused,
        recommendation_col
    )

    duplicate_savings = get_duplicate_savings(duplicates)

    st.write(
        "Estimated storage opportunities identified "
        "from duplicate detection and archive recommendations."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Storage Used",
        f"{total_size:,.2f} MB"
    )

    col2.metric(
        "Potential Duplicate Savings",
        f"{duplicate_savings:,.2f} MB"
    )

    col3.metric(
        "Archive Candidates",
        archive_count
    )

    st.metric(
        "Total Size of Archive Candidates",
        f"{archive_size:,.2f} MB"
    )

    st.info(
        "These are estimated opportunities, not confirmed savings. "
        "Archive candidates and duplicate files must be reviewed "
        "before any storage is removed."
    )


# ==========================================
# 8. GOOGLE DRIVE EXPLORER
# ==========================================

elif page == "Google Drive Explorer":

    st.title("📂 Google Drive Explorer")

    if drive.empty:
        st.warning("Google Drive data not found.")
        st.stop()

    search = st.text_input("Search file by name")

    filtered = drive.copy()

    if search and "file_name" in filtered.columns:

        filtered = filtered[
            filtered["file_name"].astype(str)
            .str.contains(search, case=False, na=False)
        ]

    if "file_type" in filtered.columns:

        types = sorted(
            filtered["file_type"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_type = st.selectbox(
            "Filter by file type",
            ["All"] + types
        )

        if selected_type != "All":

            filtered = filtered[
                filtered["file_type"] == selected_type
            ]

    st.write(
        "Total matching records:",
        len(filtered)
    )

    show_dataframe(filtered)


# ==========================================
# 9. UNUSED FILE ANALYSIS
# ==========================================

elif page == "Unused File Analysis":

    st.title("🕒 Unused File Analysis")

    if unused.empty:
        st.warning("Unused file analysis CSV not found.")
        st.stop()

    st.write(
        "Analyze Google Drive files and view "
        "rule-based storage recommendations."
    )

    recommendation_col = find_column(
        unused,
        [
            "recommendation",
            "Recommendation",
            "suggested_action",
            "status",
            "action",
            "category"
        ]
    )

    if recommendation_col:

        unused_display = unused.copy()

        unused_display[recommendation_col] = (
            unused_display[recommendation_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.title()
        )

        counts = unused_display[
            recommendation_col
        ].value_counts()

        keep_count = counts.get("Keep", 0)
        review_count = counts.get("Review", 0)
        archive_count = counts.get("Archive", 0)

        # KPI cards

        st.subheader("File Recommendation Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Keep Files", keep_count)
        col2.metric("Review Files", review_count)
        col3.metric("Archive Files", archive_count)

        st.divider()

        # Recommendation pie chart

        st.subheader("Unused File Recommendations")

        summary = (
            counts
            .rename_axis("Recommendation")
            .reset_index(name="Count")
        )

        fig = px.pie(
            summary,
            names="Recommendation",
            values="Count",
            title="File Recommendation Distribution",
            color="Recommendation",
            color_discrete_map={
                "Keep": "#60A5FA",
                "Review": "#2563EB",
                "Archive": "#EF4444",
                "Unknown": "#9CA3AF"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(fig, width="stretch")

        st.divider()

        # Archive storage summary

        st.subheader("Archive Storage Summary")

        archive_count, archive_size = get_archive_summary(
            unused_display,
            recommendation_col
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Archive Candidates",
            archive_count
        )

        col2.metric(
            "Archive Candidate Size",
            f"{archive_size:,.2f} MB"
        )

        st.divider()

        # File details

        st.subheader("File Details")

        selected_recommendation = st.selectbox(
            "Filter files by recommendation",
            ["All", "Keep", "Review", "Archive"]
        )

        filtered_unused = unused_display.copy()

        if selected_recommendation != "All":

            filtered_unused = filtered_unused[
                filtered_unused[recommendation_col]
                == selected_recommendation
            ]

        search_file = st.text_input(
            "Search by file name"
        )

        if search_file and "file_name" in filtered_unused.columns:

            filtered_unused = filtered_unused[
                filtered_unused["file_name"]
                .astype(str)
                .str.contains(
                    search_file,
                    case=False,
                    na=False
                )
            ]

        st.write(
            "Total matching files:",
            len(filtered_unused)
        )

        preferred_columns = [
            "file_name",
            "file_type",
            "file_size_mb",
            recommendation_col,
            "created_time",
            "modified_time",
            "viewed_by_me_time"
        ]

        display_columns = [
            col for col in preferred_columns
            if col in filtered_unused.columns
        ]

        show_id = st.checkbox("Show File ID")

        if show_id and "file_id" in filtered_unused.columns:
            display_columns.insert(0, "file_id")

        if display_columns:

            st.dataframe(
                readable_file_types(
                    filtered_unused[display_columns]
                ),
                width="stretch",
                hide_index=True
            )

        else:
            st.warning("No displayable columns found.")

    else:

        st.warning(
            "No recommendation column was found "
            "in the unused file analysis dataset."
        )

        show_dataframe(unused)

    st.info(
        "Recommendations are for review only. "
        "Files are not deleted automatically."
    )


# ==========================================
# 10. DUPLICATE DETECTION
# ==========================================

elif page == "Duplicate Detection":

    st.title("🔍 Duplicate File Detection")

    if duplicates.empty:
        st.warning("Duplicate file CSV not found.")
        st.stop()

    st.write(
        "Duplicate records identified using available "
        "MD5 checksums from Google Drive."
    )

    if "md5_checksum" in duplicates.columns:

        group_count = duplicates[
            "md5_checksum"
        ].dropna().nunique()

        st.metric(
            "Duplicate Groups",
            group_count
        )

    st.metric(
        "Records in Duplicate Groups",
        len(duplicates)
    )

    duplicate_savings = get_duplicate_savings(duplicates)

    st.metric(
        "Estimated Extra Storage",
        f"{duplicate_savings:,.2f} MB"
    )

    show_dataframe(duplicates)

    st.warning(
        "Duplicate detection results are for review. "
        "No files are deleted automatically."
    )


# ==========================================
# 11. AI PREDICTIONS
# ==========================================

elif page == "AI Predictions":

    st.title("🤖 Machine Learning Predictions")

    if predictions.empty:
        st.warning("ML prediction CSV not found.")
        st.stop()

    prediction_col = find_column(
        predictions,
        [
            "Prediction",
            "prediction",
            "Predicted_Label",
            "predicted_label",
            "ML_Prediction",
            "predicted_category"
        ]
    )

    confidence_col = find_column(
        predictions,
        [
            "Prediction_Confidence",
            "prediction_confidence",
            "confidence",
            "Confidence"
        ]
    )

    if prediction_col:

        st.subheader("AI Recommendation Distribution")

        summary = (
            predictions[prediction_col]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        summary.columns = ["Prediction", "Count"]

        fig = px.bar(
            summary,
            x="Prediction",
            y="Count",
            color="Prediction",
            title="ML Predictions"
        )

        st.plotly_chart(fig, width="stretch")

    if confidence_col:

        confidence = pd.to_numeric(
            predictions[confidence_col],
            errors="coerce"
        )

        if confidence.notna().any():

            st.metric(
                "Average Prediction Confidence",
                f"{confidence.mean():.2f}%"
            )

    st.info(
        "Predictions are experimental recommendations "
        "from a model trained on synthetic data. "
        "They are not verified decisions about file importance."
    )

    show_dataframe(predictions)


# ==========================================
# 12. FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "AI Cloud Storage Optimizer | MSc IT Project"
)

