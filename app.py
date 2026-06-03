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
    page_title="Inventory Duplicate Detection",
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
    "lg": "long",
    "len": "length",
    "thk": "thickness",
    "plt": "plate",
    "ctrl": "control",
    "conn": "connector",
    "matl": "material",
    "rm": "raw material",
    "fg": "finished goods",
    "wip": "work in progress",
}


def normalize_abbreviations(text: str) -> str:
    """
    Replace common inventory/material abbreviations with standardized full terms.
    """
    for short_form, full_form in ABBREVIATION_DICTIONARY.items():
        text = re.sub(
            rf"\b{re.escape(short_form)}\b",
            full_form,
            text,
            flags=re.IGNORECASE
        )
    return text


def normalize_measurements(text: str) -> str:
    """
    Normalize common material measurement patterns.
    Examples:
    M4x10 -> m4 x 10
    10MM -> 10 mm
    2M -> 2 meter
    """
    text = text.lower()

    text = re.sub(r"(\bm\d+)\s*[xX*]\s*(\d+)", r"\1 x \2", text)
    text = re.sub(r"(\d+)\s*mm\b", r"\1 mm", text)
    text = re.sub(r"(\d+)\s*cm\b", r"\1 cm", text)
    text = re.sub(r"(\d+)\s*m\b", r"\1 meter", text)
    text = re.sub(r"(\d+)\s*kg\b", r"\1 kg", text)
    text = re.sub(r"(\d+)\s*g\b", r"\1 gram", text)

    return text


def clean_inventory_description(text: str) -> str:
    """
    NLP preprocessing pipeline for inventory/material descriptions.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = normalize_abbreviations(text)
    text = normalize_measurements(text)

    # Remove special symbols but keep letters, numbers and spacing
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spacing
    text = re.sub(r"\s+", " ", text).strip()

    return text


def combine_text_columns(df: pd.DataFrame, selected_columns: list[str]) -> pd.Series:
    """
    Combine multiple selected text columns into one comparison text.
    Example: internal + external + category + type
    """
    return df[selected_columns].fillna("").astype(str).agg(" ".join, axis=1)


# ============================================================
# FILE READING
# ============================================================

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read CSV or Excel file into DataFrame.

    Supports:
    - Semicolon CSV files
    - Comma CSV files
    - Tab-separated files
    - Pipe-separated files
    - Encoding issues
    - Excel files
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

                    # Select the reading result with more columns and rows
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
# AUTO COLUMN DETECTION
# ============================================================

def auto_select_code_column(columns: list[str]) -> str:
    """
    Auto-select likely item code column.
    """
    priority = [
        "part_code",
        "item_code",
        "material_code",
        "material_no",
        "part_no",
        "item_no",
        "code",
        "id"
    ]

    lower_map = {col.lower(): col for col in columns}

    for col in priority:
        if col in lower_map:
            return lower_map[col]

    return "No item code column"


def auto_select_description_columns(columns: list[str]) -> list[str]:
    """
    Auto-select likely description columns.
    """
    priority = [
        "internal",
        "external",
        "description",
        "item_description",
        "material_description",
        "part_name",
        "brand"
    ]

    lower_map = {col.lower(): col for col in columns}
    selected = []

    for col in priority:
        if col in lower_map:
            selected.append(lower_map[col])

    if selected:
        return selected

    return columns[:1]


def auto_select_group_columns(columns: list[str]) -> list[str]:
    """
    Auto-select likely grouping columns.
    Grouping helps reduce wrong comparison.
    """
    priority = ["category", "type", "class", "process"]

    lower_map = {col.lower(): col for col in columns}
    selected = []

    for col in priority:
        if col in lower_map:
            selected.append(lower_map[col])

    return selected[:2]


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def assign_duplicate_label(
    score: float,
    duplicate_threshold: float,
    possible_threshold: float
) -> str:
    """
    Assign duplicate category based on similarity score.
    """
    if score >= duplicate_threshold:
        return "Duplicate"
    elif score >= possible_threshold:
        return "Possible Duplicate"
    return "Not Duplicate"


def validate_input_dataframe(
    df: pd.DataFrame,
    description_columns: list[str]
) -> tuple[bool, str]:
    """
    Validate uploaded inventory dataset.
    """
    if df.empty:
        return False, "The uploaded file is empty."

    if not description_columns:
        return False, "Please select at least one description column."

    for col in description_columns:
        if col not in df.columns:
            return False, f"The selected column '{col}' does not exist."

    combined_text = combine_text_columns(df, description_columns)

    if combined_text.dropna().astype(str).str.strip().eq("").all():
        return False, "The selected description columns do not contain valid text."

    return True, "Dataset is valid."


def detect_duplicate_inventory_items(
    df: pd.DataFrame,
    code_column: str | None,
    description_columns: list[str],
    group_columns: list[str],
    duplicate_threshold: float,
    possible_threshold: float,
    max_matches_per_item: int,
    compare_within_group_only: bool,
    show_not_duplicate: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect duplicate inventory/material records using:
    1. Text preprocessing
    2. TF-IDF vectorization
    3. Cosine similarity
    4. Threshold-based classification

    Detection compares one row's selected description text
    with another row's selected description text.

    The item code is used only as an identifier, not as the main similarity feature.
    """

    working_df = df.copy()

    working_df["combined_description"] = combine_text_columns(
        working_df,
        description_columns
    )

    working_df["cleaned_description"] = working_df["combined_description"].apply(
        clean_inventory_description
    )

    # Remove rows with empty cleaned descriptions
    working_df = working_df[
        working_df["cleaned_description"].str.strip() != ""
    ].reset_index(drop=True)

    if len(working_df) < 2:
        return working_df, pd.DataFrame()

    results = []

    # Compare all rows or compare only inside selected group columns
    if compare_within_group_only and group_columns:
        grouped_data = working_df.groupby(group_columns, dropna=False)
        groups = [
            (group_key, group_df.reset_index())
            for group_key, group_df in grouped_data
            if len(group_df) >= 2
        ]
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

            match_count = 0

            for j in candidate_indexes:
                if i == j:
                    continue

                original_i = int(group_df.loc[i, "index"])
                original_j = int(group_df.loc[j, "index"])

                # Avoid repeated pair A-B and B-A
                if original_i >= original_j:
                    continue

                score = float(similarity_scores[j])

                prediction = assign_duplicate_label(
                    score,
                    duplicate_threshold,
                    possible_threshold
                )

                if not show_not_duplicate and prediction == "Not Duplicate":
                    continue

                if prediction != "Not Duplicate":
                    match_count += 1

                row = {
                    "row_1": original_i + 1,
                    "row_2": original_j + 1,
                    "description_1": working_df.loc[original_i, "combined_description"],
                    "description_2": working_df.loc[original_j, "combined_description"],
                    "similarity_score": round(score, 4),
                    "similarity_percentage": round(score * 100, 2),
                    "prediction": prediction,
                    "cleaned_description_1": working_df.loc[original_i, "cleaned_description"],
                    "cleaned_description_2": working_df.loc[original_j, "cleaned_description"],
                }

                if code_column:
                    row["item_code_1"] = working_df.loc[original_i, code_column]
                    row["item_code_2"] = working_df.loc[original_j, code_column]

                for group_col in group_columns:
                    row[group_col] = working_df.loc[original_i, group_col]

                results.append(row)

                if match_count >= max_matches_per_item:
                    break

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        preferred_columns = []

        if code_column:
            preferred_columns.extend(["item_code_1", "item_code_2"])
        else:
            preferred_columns.extend(["row_1", "row_2"])

        preferred_columns.extend([
            "description_1",
            "description_2",
            "similarity_percentage",
            "prediction",
        ])

        extra_columns = [
            col for col in result_df.columns
            if col not in preferred_columns
        ]

        result_df = result_df[preferred_columns + extra_columns]
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
    """
    Convert DataFrame to downloadable CSV.
    """
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel(df: pd.DataFrame, sheet_name: str = "Results") -> bytes:
    """
    Convert DataFrame to downloadable Excel file.
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    return output.getvalue()


# ============================================================
# SAMPLE DATASET
# ============================================================

def create_sample_inventory_data() -> pd.DataFrame:
    """
    Create sample inventory/material master dataset for testing.
    """
    return pd.DataFrame({
        "part_code": [
            "MAT001", "MAT002", "MAT003", "MAT004", "MAT005",
            "MAT006", "MAT007", "MAT008", "MAT009", "MAT010",
            "MAT011", "MAT012", "MAT013", "MAT014", "MAT015",
            "MAT016", "MAT017", "MAT018", "MAT019", "MAT020"
        ],
        "internal": [
            "Screw M4 x 10mm stainless steel",
            "SS screw M4 10mm",
            "Screw M5 x 20mm stainless steel",
            "Stainless steel screw M5 20mm",
            "Rubber gasket black 20mm",
            "Black rubber seal 20 mm",
            "PCB controller board",
            "Main control PCB board",
            "Steel bracket L shape",
            "L-shaped steel mounting bracket",
            "Aluminium plate 100mm x 50mm",
            "Alum plt 100 mm x 50 mm",
            "Hex bolt M6 x 30mm",
            "Bolt hexagon M6 30 mm",
            "Plastic cover white",
            "White plastic casing cover",
            "Cable black 2 meter",
            "Cable blk 2m",
            "Sensor temperature module",
            "Temperature sensor module"
        ],
        "external": [
            "Screw stainless M4 10mm",
            "Stainless steel screw m4 x 10",
            "Screw stainless M5 20mm",
            "SS screw m5 x 20",
            "Rubber gasket black",
            "Black rubber gasket",
            "Controller PCB",
            "Main PCB controller",
            "Steel mounting bracket",
            "Mounting bracket steel L shape",
            "Aluminium flat plate",
            "Aluminium plate",
            "Hexagon bolt",
            "M6 hex bolt",
            "Plastic casing",
            "White cover casing",
            "Black cable 2m",
            "Cable black 2 meter",
            "Temperature sensor",
            "Sensor temp module"
        ],
        "uom": [
            "PCS", "PCS", "PCS", "PCS", "PCS",
            "PCS", "PCS", "PCS", "PCS", "PCS",
            "PCS", "PCS", "PCS", "PCS", "PCS",
            "PCS", "METER", "METER", "PCS", "PCS"
        ],
        "supplier": [
            "SUP A", "SUP B", "SUP A", "SUP C", "SUP D",
            "SUP E", "SUP F", "SUP G", "SUP H", "SUP I",
            "SUP J", "SUP K", "SUP L", "SUP M", "SUP N",
            "SUP O", "SUP P", "SUP Q", "SUP R", "SUP S"
        ],
        "category": [
            "Fastener", "Fastener", "Fastener", "Fastener", "Seal",
            "Seal", "Electronic", "Electronic", "Metal Part", "Metal Part",
            "Metal Part", "Metal Part", "Fastener", "Fastener", "Plastic Part",
            "Plastic Part", "Cable", "Cable", "Electronic", "Electronic"
        ],
        "type": [
            "Screw", "Screw", "Screw", "Screw", "Rubber",
            "Rubber", "PCB", "PCB", "Bracket", "Bracket",
            "Plate", "Plate", "Bolt", "Bolt", "Cover",
            "Cover", "Cable", "Cable", "Sensor", "Sensor"
        ],
        "class": [
            "General", "General", "General", "General", "General",
            "General", "Electrical", "Electrical", "Mechanical", "Mechanical",
            "Mechanical", "Mechanical", "General", "General", "General",
            "General", "Electrical", "Electrical", "Electrical", "Electrical"
        ],
        "process": [
            "PURCHASE", "PURCHASE", "PURCHASE", "PURCHASE", "PURCHASE",
            "PURCHASE", "ASSEMBLY", "ASSEMBLY", "FABRICATION", "FABRICATION",
            "FABRICATION", "FABRICATION", "PURCHASE", "PURCHASE", "INJECTION",
            "INJECTION", "ASSEMBLY", "ASSEMBLY", "ASSEMBLY", "ASSEMBLY"
        ]
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
    step=0.01,
    help="Descriptions with similarity equal or above this value will be classified as Duplicate."
)

possible_threshold = st.sidebar.slider(
    "Possible duplicate threshold",
    min_value=0.10,
    max_value=0.84,
    value=0.60,
    step=0.01,
    help="Descriptions with similarity equal or above this value will be classified as Possible Duplicate."
)

max_matches_per_item = st.sidebar.slider(
    "Maximum similar matches per item",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Limits the number of similar matches shown for each item."
)

show_not_duplicate = st.sidebar.checkbox(
    "Show Not Duplicate pairs",
    value=False,
    help="Enable only for testing because it may generate many rows."
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### NLP Method Used

    - Text preprocessing  
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
    '<div class="main-title">📦 Duplicate Inventory Item Detection Using NLP</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect duplicate or similar inventory/material master records using TF-IDF and Cosine Similarity.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    <b>Purpose:</b> This system detects duplicate inventory or material master records that may be registered
    under different item codes but have similar descriptions. The item code is used as an identifier,
    while detection is based on NLP similarity between selected description fields.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Inventory / Material Master Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your inventory/material master file in CSV or Excel format",
    type=["csv", "xlsx", "xls"]
)

use_sample_data = st.checkbox("Use sample inventory dataset instead")

df = None

if use_sample_data:
    df = create_sample_inventory_data()
    st.success("Sample inventory dataset loaded successfully.")

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
    col3.metric("Method", "TF-IDF + Cosine")

    st.dataframe(df.head(30), use_container_width=True)

    st.markdown(
        '<div class="section-title">3. Select Columns</div>',
        unsafe_allow_html=True
    )

    all_columns = df.columns.tolist()

    default_code_column = auto_select_code_column(all_columns)
    code_options = ["No item code column"] + all_columns

    selected_code_column = st.selectbox(
        "Select item/material code column",
        options=code_options,
        index=code_options.index(default_code_column) if default_code_column in code_options else 0
    )

    default_description_columns = auto_select_description_columns(all_columns)

    selected_description_columns = st.multiselect(
        "Select description column(s) for NLP comparison",
        options=all_columns,
        default=default_description_columns,
        help="For your dataset, recommended columns are internal and external. You may also include brand if useful."
    )

    default_group_columns = auto_select_group_columns(all_columns)

    selected_group_columns = st.multiselect(
        "Optional: compare only within same group/category",
        options=all_columns,
        default=default_group_columns,
        help="Recommended: category and type. This reduces wrong matches between unrelated items."
    )

    compare_within_group_only = st.checkbox(
        "Only compare rows inside the same selected group/category",
        value=True,
        help="If enabled, items are compared only when they belong to the same selected category/type/class/process."
    )

    code_column = None if selected_code_column == "No item code column" else selected_code_column

    is_valid, validation_message = validate_input_dataframe(
        df,
        selected_description_columns
    )

    if is_valid:
        st.markdown(
            f"""
            <div class="success-box">
            <b>Validation:</b> {validation_message}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="warning-box">
            <b>Validation:</b> {validation_message}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">4. Run Duplicate Detection</div>',
        unsafe_allow_html=True
    )

    run_detection = st.button(
        "🚀 Detect Duplicate Inventory Items",
        type="primary"
    )

    if run_detection:

        if possible_threshold >= duplicate_threshold:
            st.error(
                "Please adjust the thresholds. Possible duplicate threshold must be lower than duplicate threshold."
            )

        elif not is_valid:
            st.error(validation_message)

        else:
            with st.spinner(
                "Cleaning descriptions, creating TF-IDF vectors, and calculating similarity..."
            ):
                cleaned_df, result_df = detect_duplicate_inventory_items(
                    df=df,
                    code_column=code_column,
                    description_columns=selected_description_columns,
                    group_columns=selected_group_columns,
                    duplicate_threshold=duplicate_threshold,
                    possible_threshold=possible_threshold,
                    max_matches_per_item=max_matches_per_item,
                    compare_within_group_only=compare_within_group_only,
                    show_not_duplicate=show_not_duplicate
                )

            st.markdown(
                '<div class="section-title">5. Cleaned Text Output</div>',
                unsafe_allow_html=True
            )

            preview_columns = selected_description_columns + [
                "combined_description",
                "cleaned_description"
            ]

            st.dataframe(
                cleaned_df[preview_columns].head(50),
                use_container_width=True
            )

            st.markdown(
                '<div class="section-title">6. Detection Summary</div>',
                unsafe_allow_html=True
            )

            if result_df.empty:
                st.warning(
                    "No duplicate or possible duplicate inventory items were detected. Try lowering the threshold."
                )
            else:
                duplicate_count = len(
                    result_df[result_df["prediction"] == "Duplicate"]
                )

                possible_count = len(
                    result_df[result_df["prediction"] == "Possible Duplicate"]
                )

                not_duplicate_count = len(
                    result_df[result_df["prediction"] == "Not Duplicate"]
                )

                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Duplicate", duplicate_count)
                m2.metric("Possible Duplicate", possible_count)
                m3.metric("Not Duplicate", not_duplicate_count)
                m4.metric("Total Compared Pairs", len(result_df))

                summary_df = result_df["prediction"].value_counts().reset_index()
                summary_df.columns = ["Prediction", "Count"]

                fig, ax = plt.subplots()
                ax.bar(summary_df["Prediction"], summary_df["Count"])
                ax.set_xlabel("Prediction Category")
                ax.set_ylabel("Number of Item Pairs")
                ax.set_title("Inventory Duplicate Detection Summary")
                plt.xticks(rotation=20)

                st.pyplot(fig)

                st.markdown(
                    '<div class="section-title">7. Duplicate Detection Results</div>',
                    unsafe_allow_html=True
                )

                filter_option = st.selectbox(
                    "Filter result by prediction",
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

                download_col1, download_col2 = st.columns(2)

                with download_col1:
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=dataframe_to_csv(filtered_result_df),
                        file_name="inventory_duplicate_detection_results.csv",
                        mime="text/csv"
                    )

                with download_col2:
                    st.download_button(
                        label="📥 Download Results as Excel",
                        data=dataframe_to_excel(
                            filtered_result_df,
                            "Duplicate Results"
                        ),
                        file_name="inventory_duplicate_detection_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                st.markdown(
                    '<div class="section-title">9. NLP Method Explanation</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    This system uses the following NLP workflow:

                    1. **Text Preprocessing**  
                       Converts inventory/material descriptions into lowercase, removes symbols, and cleans extra spacing.

                    2. **Abbreviation Normalization**  
                       Converts common inventory abbreviations such as `SS`, `PCB`, `BLK`, and `BRKT` into standardized terms.

                    3. **TF-IDF Vectorization**  
                       Converts cleaned item descriptions into numerical vectors based on word importance.

                    4. **Cosine Similarity**  
                       Measures how similar one inventory item description is to another inventory item description.

                    5. **Threshold-Based Classification**  
                       Classifies item pairs as `Duplicate`, `Possible Duplicate`, or `Not Duplicate`.

                    **Important:** The item/material code is not used to calculate similarity. It is only used as an identifier in the result table.
                    """
                )

else:
    st.markdown(
        """
        <div class="warning-box">
        Please upload a CSV or Excel inventory/material master file, or select the sample dataset option.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Sample Inventory Dataset Format</div>',
        unsafe_allow_html=True
    )

    sample_df = create_sample_inventory_data()
    st.dataframe(sample_df, use_container_width=True)

    st.download_button(
        label="📥 Download Sample Inventory CSV",
        data=dataframe_to_csv(sample_df),
        file_name="sample_inventory_dataset.csv",
        mime="text/csv"
    )
