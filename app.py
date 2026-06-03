import re
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


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
# TEXT PREPROCESSING
# ============================================================

ABBREVIATION_DICTIONARY = {
    "ss": "stainless steel",
    "s/s": "stainless steel",
    "stn stl": "stainless steel",
    "stainless": "stainless steel",
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
    "assy": "assembly",
}


def normalize_abbreviations(text: str) -> str:
    """
    Replace common BOM abbreviations with standardized full terms.
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
    Normalize common BOM measurement patterns.
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

    return text


def clean_bom_description(text: str) -> str:
    """
    Full NLP preprocessing pipeline for BOM item descriptions.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = normalize_abbreviations(text)
    text = normalize_measurements(text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# DUPLICATE DETECTION FUNCTIONS
# ============================================================

def assign_duplicate_label(score: float, duplicate_threshold: float, possible_threshold: float) -> str:
    """
    Assign duplicate category based on similarity score.
    """
    if score >= duplicate_threshold:
        return "Duplicate"
    elif score >= possible_threshold:
        return "Possible Duplicate"
    return "Not Duplicate"


def validate_input_dataframe(df: pd.DataFrame, description_column: str) -> tuple[bool, str]:
    """
    Validate uploaded BOM dataset.
    """
    if df.empty:
        return False, "The uploaded file is empty."

    if description_column not in df.columns:
        return False, "The selected description column does not exist."

    if df[description_column].dropna().empty:
        return False, "The selected description column does not contain valid text."

    return True, "Dataset is valid."


def detect_duplicate_bom_items(
    df: pd.DataFrame,
    description_column: str,
    code_column: str | None,
    duplicate_threshold: float,
    possible_threshold: float,
    show_not_duplicate: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect duplicate BOM item descriptions using:
    1. Text preprocessing
    2. TF-IDF vectorization
    3. Cosine similarity
    4. Threshold-based classification
    """

    working_df = df.copy()

    working_df[description_column] = working_df[description_column].fillna("").astype(str)
    working_df["cleaned_description"] = working_df[description_column].apply(clean_bom_description)

    working_df = working_df[
        working_df["cleaned_description"].str.strip() != ""
    ].reset_index(drop=True)

    if len(working_df) < 2:
        return working_df, pd.DataFrame()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(working_df["cleaned_description"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results = []

    for i in range(len(working_df)):
        for j in range(i + 1, len(working_df)):
            similarity_score = similarity_matrix[i][j]

            predicted_label = assign_duplicate_label(
                similarity_score,
                duplicate_threshold,
                possible_threshold
            )

            if not show_not_duplicate and predicted_label == "Not Duplicate":
                continue

            row = {
                "item_1_row": i + 1,
                "item_2_row": j + 1,
                "description_1": working_df.loc[i, description_column],
                "description_2": working_df.loc[j, description_column],
                "cleaned_description_1": working_df.loc[i, "cleaned_description"],
                "cleaned_description_2": working_df.loc[j, "cleaned_description"],
                "similarity_score": round(similarity_score, 4),
                "similarity_percentage": round(similarity_score * 100, 2),
                "prediction": predicted_label
            }

            if code_column:
                row["item_code_1"] = working_df.loc[i, code_column]
                row["item_code_2"] = working_df.loc[j, code_column]

            results.append(row)

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        preferred_columns = []

        if code_column:
            preferred_columns.extend(["item_code_1", "item_code_2"])
        else:
            preferred_columns.extend(["item_1_row", "item_2_row"])

        preferred_columns.extend([
            "description_1",
            "description_2",
            "similarity_percentage",
            "prediction",
            "cleaned_description_1",
            "cleaned_description_2",
        ])

        result_df = result_df[preferred_columns]
        result_df = result_df.sort_values(
            by="similarity_percentage",
            ascending=False
        ).reset_index(drop=True)

    return working_df, result_df


# ============================================================
# FILE HANDLING
# ============================================================

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read CSV or Excel file into DataFrame.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or Excel file.")


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to downloadable CSV.
    """
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel(df: pd.DataFrame, sheet_name: str = "Results") -> bytes:
    """
    Convert DataFrame to downloadable Excel file.
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    return output.getvalue()


# ============================================================
# OPTIONAL EVALUATION
# ============================================================

def convert_label_to_binary(label: str) -> int:
    """
    Convert labels to binary format for evaluation.
    Duplicate and Possible Duplicate are considered duplicate class.
    """
    label = str(label).strip().lower()

    if label in ["duplicate", "possible duplicate", "yes", "1", "true"]:
        return 1

    return 0


def evaluate_predictions(result_df: pd.DataFrame, actual_label_column: str) -> dict:
    """
    Calculate evaluation metrics if actual labels are provided.
    """
    y_true = result_df[actual_label_column].apply(convert_label_to_binary)
    y_pred = result_df["prediction"].apply(convert_label_to_binary)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "report": classification_report(y_true, y_pred, zero_division=0)
    }

    return metrics


# ============================================================
# SAMPLE DATA
# ============================================================

def create_sample_bom_data() -> pd.DataFrame:
    """
    Create sample BOM dataset for user testing.
    """
    return pd.DataFrame({
        "item_code": [
            "BOM001",
            "BOM002",
            "BOM003",
            "BOM004",
            "BOM005",
            "BOM006",
            "BOM007",
            "BOM008",
            "BOM009",
            "BOM010"
        ],
        "description": [
            "Screw M4 x 10mm stainless steel",
            "SS screw M4 10mm",
            "Rubber gasket black 20mm",
            "Black rubber seal 20mm",
            "PCB controller board",
            "Main control PCB board",
            "Steel bracket L shape",
            "L-shaped steel mounting bracket",
            "Cable black 2 meter",
            "Black cable 2m"
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
    value=0.80,
    step=0.01,
    help="Descriptions with similarity equal or above this value will be classified as Duplicate."
)

possible_threshold = st.sidebar.slider(
    "Possible duplicate threshold",
    min_value=0.10,
    max_value=0.79,
    value=0.50,
    step=0.01,
    help="Descriptions with similarity equal or above this value will be classified as Possible Duplicate."
)

show_not_duplicate = st.sidebar.checkbox(
    "Show Not Duplicate pairs",
    value=False,
    help="Enable this only for testing because it may produce many results."
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### NLP Method Used

    **Text preprocessing**  
    **Abbreviation normalization**  
    **TF-IDF vectorization**  
    **Cosine similarity**  
    **Threshold-based classification**
    """
)

if possible_threshold >= duplicate_threshold:
    st.sidebar.error("Possible duplicate threshold must be lower than duplicate threshold.")


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown('<div class="main-title">📦 Duplicate BOM Item Detection Using NLP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Detect duplicate or similar Bill of Materials item descriptions using TF-IDF and Cosine Similarity.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    <b>Purpose:</b> This system helps identify duplicate BOM item descriptions that may be written in different formats,
    such as <i>"SS screw M4 10mm"</i> and <i>"Screw M4 x 10mm stainless steel"</i>.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown('<div class="section-title">1. Upload BOM Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your BOM file in CSV or Excel format",
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

    st.markdown('<div class="section-title">2. Dataset Preview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Duplicate Check Method", "TF-IDF + Cosine")

    st.dataframe(df.head(30), use_container_width=True)

    st.markdown('<div class="section-title">3. Select Columns</div>', unsafe_allow_html=True)

    all_columns = df.columns.tolist()

    selected_code_column = st.selectbox(
        "Select item code column",
        options=["No item code column"] + all_columns
    )

    selected_description_column = st.selectbox(
        "Select BOM description column",
        options=all_columns
    )

    code_column = None if selected_code_column == "No item code column" else selected_code_column

    is_valid, validation_message = validate_input_dataframe(df, selected_description_column)

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

    st.markdown('<div class="section-title">4. Run Duplicate Detection</div>', unsafe_allow_html=True)

    run_detection = st.button("🚀 Detect Duplicate BOM Items", type="primary")

    if run_detection:

        if possible_threshold >= duplicate_threshold:
            st.error("Please adjust the thresholds. Possible duplicate threshold must be lower than duplicate threshold.")

        elif not is_valid:
            st.error(validation_message)

        else:
            with st.spinner("Cleaning descriptions, creating TF-IDF vectors, and calculating similarity..."):
                cleaned_df, result_df = detect_duplicate_bom_items(
                    df=df,
                    description_column=selected_description_column,
                    code_column=code_column,
                    duplicate_threshold=duplicate_threshold,
                    possible_threshold=possible_threshold,
                    show_not_duplicate=show_not_duplicate
                )

            st.markdown('<div class="section-title">5. Cleaned Text Output</div>', unsafe_allow_html=True)

            st.dataframe(
                cleaned_df[[selected_description_column, "cleaned_description"]],
                use_container_width=True
            )

            st.markdown('<div class="section-title">6. Detection Summary</div>', unsafe_allow_html=True)

            if result_df.empty:
                st.warning("No duplicate or possible duplicate BOM items were detected.")
            else:
                duplicate_count = len(result_df[result_df["prediction"] == "Duplicate"])
                possible_count = len(result_df[result_df["prediction"] == "Possible Duplicate"])
                not_duplicate_count = len(result_df[result_df["prediction"] == "Not Duplicate"])

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
                ax.set_title("BOM Duplicate Detection Summary")
                plt.xticks(rotation=20)

                st.pyplot(fig)

                st.markdown('<div class="section-title">7. Duplicate Detection Results</div>', unsafe_allow_html=True)

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

                st.markdown('<div class="section-title">8. Download Results</div>', unsafe_allow_html=True)

                download_col1, download_col2 = st.columns(2)

                with download_col1:
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=dataframe_to_csv(filtered_result_df),
                        file_name="bom_duplicate_detection_results.csv",
                        mime="text/csv"
                    )

                with download_col2:
                    st.download_button(
                        label="📥 Download Results as Excel",
                        data=dataframe_to_excel(filtered_result_df, "Duplicate Results"),
                        file_name="bom_duplicate_detection_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                st.markdown('<div class="section-title">9. Method Explanation</div>', unsafe_allow_html=True)

                st.markdown(
                    """
                    This system uses the following NLP workflow:

                    1. **Text preprocessing**  
                       The BOM description is converted into lowercase, symbols are removed, and spacing is cleaned.

                    2. **Abbreviation normalization**  
                       Common manufacturing abbreviations such as `SS`, `PCB`, `BLK`, and `BRKT` are converted into standard words.

                    3. **TF-IDF vectorization**  
                       The cleaned BOM descriptions are converted into numerical vectors based on word importance.

                    4. **Cosine similarity**  
                       Each BOM item description is compared with another description to calculate similarity.

                    5. **Threshold-based classification**  
                       The system classifies each pair as `Duplicate`, `Possible Duplicate`, or `Not Duplicate`.
                    """
                )

else:
    st.markdown(
        """
        <div class="warning-box">
        Please upload a CSV or Excel BOM file, or select the sample dataset option.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Sample BOM Dataset Format</div>', unsafe_allow_html=True)

    sample_df = create_sample_bom_data()
    st.dataframe(sample_df, use_container_width=True)

    st.download_button(
        label="📥 Download Sample BOM CSV",
        data=dataframe_to_csv(sample_df),
        file_name="sample_bom_dataset.csv",
        mime="text/csv"
    )
