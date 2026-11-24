
import pandas as pd


def analyze_unused_files(input_df=None):

    # ==========================================
    # 1. LOAD GOOGLE DRIVE DATA
    # ==========================================

    if input_df is None:
        df = pd.read_csv("google_drive_files.csv")
    else:
        df = input_df.copy()

    print("\nGoogle Drive data loaded successfully!")
    print("Total files:", len(df))

    # ==========================================
    # 2. CONVERT DATE COLUMNS
    # ==========================================

    date_columns = [
        "created_time",
        "modified_time",
        "viewed_by_me_time"
    ]

    for column in date_columns:

        if column not in df.columns:
            df[column] = pd.NaT

        df[column] = pd.to_datetime(
            df[column],
            utc=True,
            errors="coerce"
        )

    # ==========================================
    # 3. CALCULATE FILE ACTIVITY
    # ==========================================

    today = pd.Timestamp.now(tz="UTC")

    df["file_age_days"] = (
        today - df["created_time"]
    ).dt.days

    df["last_modified_days"] = (
        today - df["modified_time"]
    ).dt.days

    df["days_since_viewed"] = (
        today - df["viewed_by_me_time"]
    ).dt.days

    # ==========================================
    # 4. CHECK STARRED FILES
    # ==========================================

    def is_file_starred(value):

        return str(value).strip().lower() in [
            "true", "1", "yes"
        ]

    # ==========================================
    # 5. FILE RECOMMENDATION LOGIC
    # ==========================================

    def recommend_file(row):

        # Protect starred files
        if is_file_starred(
            row.get("is_starred", False)
        ):
            return "Keep"

        # Unknown viewing history
        if pd.isna(row["days_since_viewed"]):
            return "Review"

        # Recently viewed files
        if row["days_since_viewed"] <= 90:
            return "Keep"

        # Old files with no recent activity
        if (
            row["days_since_viewed"] > 730
            and row["last_modified_days"] > 730
        ):
            return "Archive"

        # Not viewed for more than one year
        if row["days_since_viewed"] > 365:
            return "Review"

        return "Keep"

    df["recommendation"] = df.apply(
        recommend_file,
        axis=1
    )

    # ==========================================
    # 6. SAVE CSV FILE
    # ==========================================

    df.to_csv(
        "unused_file_analysis.csv",
        index=False
    )

    # ==========================================
    # 7. SAVE EXCEL FILE
    # ==========================================

    excel_df = df.copy()

    # Excel does not support timezone-aware dates.
    # Remove timezone information only in Excel copy.

    for column in date_columns:

        if column in excel_df.columns:

            excel_df[column] = (
                excel_df[column].dt.tz_localize(None)
            )

    excel_df.to_excel(
        "unused_file_analysis.xlsx",
        index=False
    )

    print("\nCSV file saved successfully!")
    print("Excel file saved successfully!")

    # ==========================================
    # 8. DISPLAY FILE ANALYSIS
    # ==========================================

    print("\n========== UNUSED FILE ANALYSIS ==========")

    display_columns = [
        "file_name",
        "file_size_mb",
        "file_age_days",
        "last_modified_days",
        "days_since_viewed",
        "recommendation"
    ]

    print(
        df[display_columns]
        .head(20)
        .to_string(index=False)
    )

    # ==========================================
    # 9. RECOMMENDATION SUMMARY
    # ==========================================

    print("\n========== RECOMMENDATION SUMMARY ==========")

    summary = (
        df["recommendation"]
        .value_counts()
    )

    print(summary)

    print("\nTotal analysed files:", len(df))

    print("\nGenerated files:")
    print("1. unused_file_analysis.csv")
    print("2. unused_file_analysis.xlsx")

    return df


# ==========================================
# 10. RUN THE PROGRAM
# ==========================================

if __name__ == "__main__":

    analyze_unused_files()
