import re
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BOM Duplicate Detection",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #6B7280;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .info-box {
        background-color: #F3F4F6;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 20px;
    }

    .success-box {
        background-color: #ECFDF5;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin-bottom: 20px;
    }

    .warning-box {
        background-color: #FFFBEB;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NLP PREPROCESSING
# ============================================================

ABBREVIATION_DICTIONARY = {
    "ss": "stainless steel",
    "s/s": "stainless steel",
    "stn stl": "stainless steel",
    "blk": "black",
    "bk": "black",
    "wht": "white",
    "gry": "grey",
    "assy": "assembly",
    "asm": "assembly",
    "pcb": "printed circuit board",
    "pcba": "printed circuit board assembly",
    "qty": "quantity",
    "alum": "aluminium",
    "alu": "aluminium",
    "brkt": "bracket",
    "scrw": "screw",
    "scr": "screw",
    "hd": "head",
    "hex": "hexagon",
    "dia": "diameter",
    "od": "outer diameter",
    "id": "inner diameter",
    "mtr": "meter",
    "mt": "meter",
    "ctrl": "control",
    "conn": "connector",
    "fix": "fixing",
    "stp": "stamping",
}


def normalize_abbreviations(text: str) -> str:
    for short_form, full_form in ABBREVIATION_DICTIONARY.items():
        text = re.sub(
            rf"\b{re.escape(short_form)}\b",
            full_form,
            text,
            flags=re.IGNORECASE
        )
    return text


def normalize_measurements(text: str) -> str:
    text = text.lower()

    # M4x10, M4 X 10, M4*10 -> m4 x 10
    text = re.sub(r"(\bm\d+)\s*[xX*]\s*(\d+)", r"\1 x \2", text)

    # 10MM -> 10 mm
    text = re.sub(r"(\d+)\s*mm\b", r"\1 mm", text)

    # 2M -> 2 meter
    text = re.sub(r"(\d+)\s*m\b", r"\1 meter", text)

    return text


def clean_bom_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = normalize_abbreviations(text)
    text = normalize_measurements(text)

    # Keep letters and numbers only
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def combine_text_columns(df: pd.DataFrame, selected_columns: list[str]) -> pd.Series:
    combined = df[selected_columns].fillna("").astype(str).agg(" ".join, axis=1)
    return combined


# ============================================================
# FILE READING
# ============================================================

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read CSV or Excel file.
    Supports your code_data.csv format:
    - semicolon separator
    - quoted values
    - latin1/cp1252 encoding
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        separators = [";", ",", "\t", "|", None]
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

        best_df = None
        best_score = -1
        last_error = None

        for encoding in encodings:
            for separator in separators:
                try:
                    uploaded_file.seek(0)

                    df = pd.read_csv(
                        uploaded_file,
                        sep=separator,
                        engine="python",
                        encoding=encoding,
                        quotechar='"',
                        on_bad_lines="skip"
                    )

                    if df.empty:
                        continue

                    # Choose the read result with the most columns and rows
                    score = len(df.columns) * 1000 + len(df)

                    if score > best_score:
                        best_score = score
                        best_df = df

                except Exception as error:
                    last_error = error

        if best_df is not None:
            return best_df

        raise ValueError(f"Unable to read CSV file. Last error: {last_error}")

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or Excel file.")


# ============================================================
# COLUMN AUTO-DETECTION
# ============================================================

def auto_select_code_column(columns: list[str]) -> str:
    priority = ["part_code", "item_code", "code", "part_no", "item_no", "id"]

    lower_map = {col.lower(): col for col in columns}

    for col in priority:
        if col in lower_map:
            return lower_map[col]

    return "No item code column"


def auto_select_description_columns(columns: list[str]) -> list[str]:
    priority = ["internal", "description", "item_description", "part_name", "external"]

    lower_map = {col.lower(): col for col in columns}
    selected = []

    for col in priority:
        if col in lower_map:
            selected.append(lower_map[col])

    if selected:
        return selected

    return columns[:1]


def get_existing_columns(columns: list[str], wanted: list[str]) -> list[str]:
    lower_map = {col.lower(): col for col in columns}
    found = []

    for col in wanted:
        if col.lower() in lower_map:
            found.append(lower_map[col.lower()])

    return found


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def assign_duplicate_label(
    score: float,
    duplicate_threshold: float,
    possible_threshold: float
) -> str:
    if score >= duplicate_threshold:
        return "Duplicate"
    elif score >= possible_threshold:
        return "Possible Duplicate"
    return "Not Duplicate"


def detect_duplicate_bom_items(
    df: pd.DataFrame,
    code_column: str | None,
    description_columns: list[str],
    block_columns: list[str],
    duplicate_threshold: float,
    possible_threshold: float,
    top_n: int,
    compare_within_same_block_only: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:

    working_df = df.copy()

    working_df["combined_description"] = combine_text_columns(
        working_df,
        description_columns
    )

    working_df["cleaned_description"] = working_df["combined_description"].apply(
        clean_bom_text
    )

    working_df = working_df[
        working_df["cleaned_description"].str.strip() != ""
    ].reset_index(drop=True)

    if len(working_df) < 2:
        return working_df, pd.DataFrame()

    results = []

    if compare_within_same_block_only and block_columns:
        grouped_data = working_df.groupby(block_columns, dropna=False)
        groups = [(group_key, group_df.reset_index()) for group_key, group_df in grouped_data]
    else:
        groups = [("All Data", working_df.reset_index())]

    for group_key, group_df in groups:
        if len(group_df) < 2:
            continue

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1
        )

        tfidf_matrix = vectorizer.fit_transform(group_df["cleaned_description"])
        similarity_matrix = cosine_similarity(tfidf_matrix)

        for i in range(len(group_df)):
            similarity_scores = similarity_matrix[i]

            candidate_indexes = similarity_scores.argsort()[::-1]

            added_count = 0

            for j in candidate_indexes:
                if i == j:
                    continue

                original_i = int(group_df.loc[i, "index"])
                original_j = int(group_df.loc[j, "index"])

                # Avoid duplicate pair: A-B and B-A
                if original_i >= original_j:
                    continue

                score = float(similarity_scores[j])

                if score < possible_threshold:
                    continue

                prediction = assign_duplicate_label(
                    score,
                    duplicate_threshold,
                    possible_threshold
                )

                row = {
                    "row_1": original_i + 1,
                    "row_2": original_j + 1,
                    "description_1": working_df.loc[original_i, "combined_description"],
                    "description_2": working_df.loc[original_j, "combined_description"],
                    "cleaned_description_1": working_df.loc[original_i, "cleaned_description"],
                    "cleaned_description_2": working_df.loc[original_j, "cleaned_description"],
                    "similarity_percentage": round(score * 100, 2),
                    "prediction": prediction,
                }

                if code_column:
                    row["item_code_1"] = working_df.loc[original_i, code_column]
                    row["item_code_2"] = working_df.loc[original_j, code_column]

                for block_col in block_columns:
                    row[block_col] = working_df.loc[original_i, block_col]

                results.append(row)

                added_count += 1

                if added_count >= top_n:
                    break

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        front_columns = []

        if code_column:
            front_columns.extend(["item_code_1", "item_code_2"])
        else:
            front_columns.extend(["row_1", "row_2"])

        front_columns.extend([
            "description_1",
            "description_2",
            "similarity_percentage",
            "prediction",
        ])

        extra_columns = [
            col for col in result_df.columns
            if col not in front_columns
        ]

        result_df = result_df[front_columns + extra_columns]
        result_df = result_df.drop_duplicates()
        result_df = result_df.sort_values(
            by="similarity_percentage",
            ascending=False
        ).reset_index(drop=True)

    return working_df, result_df


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel(df: pd.DataFrame, sheet_name: str = "Results") -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    return output.getvalue()


# ============================================================
# SAMPLE DATA
# ============================================================

def create_sample_bom_data() -> pd.DataFrame:
    return pd.DataFrame({
        "part_code": [
            "BOM001", "BOM002", "BOM003", "BOM004", "BOM005",
            "BOM006", "BOM007", "BOM008", "BOM009", "BOM010"
        ],
        "internal": [
            "Screw M4 x 10mm stainless steel",
            "SS screw M4 10mm",
            "Rubber gasket black 20mm",
            "Black rubber seal 20 mm",
            "PCB controller board",
            "Main control PCB board",
            "Steel bracket L shape",
            "L-shaped steel mounting bracket",
            "Cable black 2 meter",
            "Cable blk 2m"
        ],
        "category": [
            "Fastener", "Fastener", "Seal", "Seal", "Electronic",
            "Electronic", "Metal Part", "Metal Part", "Cable", "Cable"
        ],
        "type": [
            "Screw", "Screw", "Rubber", "Rubber", "PCB",
            "PCB", "Bracket", "Bracket", "Cable", "Cable"
        ],
    })


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("### Similarity Thresholds")

duplicate_threshold = st.sidebar.slider(
    "Duplicate threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.85,
    step=0.01
)

possible_threshold = st.sidebar.slider(
    "Possible duplicate threshold",
    min_value=0.10,
    max_value=0.84,
    value=0.60,
    step=0.01
)

top_n = st.sidebar.slider(
    "Maximum similar matches per item",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### NLP Components

    - Text cleaning
    - Abbreviation normalization
    - TF-IDF vectorization
    - Cosine similarity
    - Threshold-based classification
    """
)

if possible_threshold >= duplicate_threshold:
    st.sidebar.error("Possible duplicate threshold must be lower than duplicate threshold.")


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown(
    '<div class="main-title">📦 Duplicate BOM Item Detection Using NLP</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Compare one BOM row with another BOM row using item descriptions, TF-IDF, and Cosine Similarity.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    <b>How detection works:</b> The system compares <b>one row description</b> with
    <b>another row description</b>. The item code is used only as an identifier in the output.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload BOM Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload BOM file in CSV or Excel format",
    type=["csv", "xlsx", "xls"]
)

use_sample_data = st.checkbox("Use sample BOM dataset instead")

df = None

if use_sample_data:
    df = create_sample_bom_data()
    st.success("Sample BOM dataset loaded successfully.")

elif uploaded_file is not None:
    try:
        df = read_uploaded_file(uploaded_file)
        st.success("File uploaded successfully.")
    except Exception as error:
        st.error(f"Failed to read file: {error}")


# ============================================================
# PROCESS DATASET
# ============================================================

if df is not None:

    st.markdown(
        '<div class="section-title">2. Dataset Preview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Detection Method", "TF-IDF + Cosine")

    st.dataframe(df.head(30), use_container_width=True)

    st.markdown(
        '<div class="section-title">3. Select Columns</div>',
        unsafe_allow_html=True
    )

    all_columns = df.columns.tolist()

    default_code = auto_select_code_column(all_columns)
    default_description_columns = auto_select_description_columns(all_columns)

    code_options = ["No item code column"] + all_columns

    selected_code_column = st.selectbox(
        "Select item code column",
        options=code_options,
        index=code_options.index(default_code) if default_code in code_options else 0
    )

    selected_description_columns = st.multiselect(
        "Select description column(s) used for NLP comparison",
        options=all_columns,
        default=default_description_columns,
        help="For your code_data.csv, recommended: part_code, internal, external. If external has many blanks, use part_code + internal."
    )

    recommended_block_columns = get_existing_columns(
        all_columns,
        ["category", "type", "class", "process"]
    )

    selected_block_columns = st.multiselect(
        "Optional: compare only within same group/category",
        options=all_columns,
        default=get_existing_columns(all_columns, ["category", "type"]),
        help="This reduces wrong matches. For your file, category/type is useful."
    )

    compare_within_same_block_only = st.checkbox(
        "Only compare rows inside the same selected group/category",
        value=True
    )

    code_column = None if selected_code_column == "No item code column" else selected_code_column

    if not selected_description_columns:
        st.warning("Please select at least one description column.")

    st.markdown(
        '<div class="section-title">4. Run Duplicate Detection</div>',
        unsafe_allow_html=True
    )

    run_detection = st.button(
        "🚀 Detect Duplicate BOM Items",
        type="primary"
    )

    if run_detection:

        if possible_threshold >= duplicate_threshold:
            st.error("Possible duplicate threshold must be lower than duplicate threshold.")

        elif not selected_description_columns:
            st.error("Please select at least one description column.")

        else:
            with st.spinner("Running NLP duplicate detection..."):
                cleaned_df, result_df = detect_duplicate_bom_items(
                    df=df,
                    code_column=code_column,
                    description_columns=selected_description_columns,
                    block_columns=selected_block_columns,
                    duplicate_threshold=duplicate_threshold,
                    possible_threshold=possible_threshold,
                    top_n=top_n,
                    compare_within_same_block_only=compare_within_same_block_only
                )

            st.markdown(
                '<div class="section-title">5. Cleaned Text Output</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                cleaned_df[
                    selected_description_columns + ["combined_description", "cleaned_description"]
                ].head(50),
                use_container_width=True
            )

            st.markdown(
                '<div class="section-title">6. Detection Summary</div>',
                unsafe_allow_html=True
            )

            if result_df.empty:
                st.warning("No duplicate or possible duplicate items found. Try lowering the threshold.")
            else:
                duplicate_count = len(result_df[result_df["prediction"] == "Duplicate"])
                possible_count = len(result_df[result_df["prediction"] == "Possible Duplicate"])

                m1, m2, m3 = st.columns(3)
                m1.metric("Duplicate", duplicate_count)
                m2.metric("Possible Duplicate", possible_count)
                m3.metric("Total Similar Pairs", len(result_df))

                summary_df = result_df["prediction"].value_counts().reset_index()
                summary_df.columns = ["Prediction", "Count"]

                fig, ax = plt.subplots()
                ax.bar(summary_df["Prediction"], summary_df["Count"])
                ax.set_xlabel("Prediction")
                ax.set_ylabel("Number of Pairs")
                ax.set_title("Duplicate Detection Summary")
                plt.xticks(rotation=15)
                st.pyplot(fig)

                st.markdown(
                    '<div class="section-title">7. Duplicate Detection Results</div>',
                    unsafe_allow_html=True
                )

                filter_option = st.selectbox(
                    "Filter by prediction",
                    options=["All"] + sorted(result_df["prediction"].unique().tolist())
                )

                filtered_result_df = result_df.copy()

                if filter_option != "All":
                    filtered_result_df = filtered_result_df[
                        filtered_result_df["prediction"] == filter_option
                    ]

                st.dataframe(filtered_result_df, use_container_width=True)

                st.markdown(
                    '<div class="section-title">8. Download Results</div>',
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.download_button(
                        label="📥 Download CSV",
                        data=dataframe_to_csv(filtered_result_df),
                        file_name="bom_duplicate_results.csv",
                        mime="text/csv"
                    )

                with c2:
                    st.download_button(
                        label="📥 Download Excel",
                        data=dataframe_to_excel(filtered_result_df),
                        file_name="bom_duplicate_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                st.markdown(
                    '<div class="section-title">9. Method Explanation</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    **How the system detects duplicates:**

                    The system does **not** detect duplicates using item code directly.  
                    It compares the selected text description columns from one row with another row.

                    Example for your dataset:

                    - `part_code` is used as the item identifier
                    - `internal` / `external` are used as description text
                    - `category` / `type` can be used to limit comparison within the same group

                    NLP workflow:

                    1. Combine selected description columns  
                    2. Clean and normalize the text  
                    3. Convert text into TF-IDF vectors  
                    4. Compare row-to-row similarity using Cosine Similarity  
                    5. Classify the pair as Duplicate or Possible Duplicate
                    """
                )

else:
    st.markdown(
        """
        <div class="warning-box">
        Please upload a CSV or Excel BOM file, or use the sample dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    sample_df = create_sample_bom_data()

    st.markdown(
        '<div class="section-title">Sample Dataset Format</div>',
        unsafe_allow_html=True
    )

    st.dataframe(sample_df, use_container_width=True)

    st.download_button(
        label="📥 Download Sample CSV",
        data=dataframe_to_csv(sample_df),
        file_name="sample_bom_dataset.csv",
        mime="text/csv"
    )
