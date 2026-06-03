import re
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Inventory Duplicate Detector",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

default_states = {
    "result_df": pd.DataFrame(),
    "technical_result_df": pd.DataFrame(),
    "cleaned_df": pd.DataFrame(),
    "detection_done": False,
    "selected_code_column": None,
    "selected_description_column": None,
    "source_key": None,
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_detection_state():
    st.session_state.result_df = pd.DataFrame()
    st.session_state.technical_result_df = pd.DataFrame()
    st.session_state.cleaned_df = pd.DataFrame()
    st.session_state.detection_done = False
    st.session_state.selected_code_column = None
    st.session_state.selected_description_column = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #6B7280;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .info-box {
        background-color: #F3F4F6;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #2563EB;
        margin-bottom: 22px;
        color: #1F2937;
    }

    .method-box {
        background-color: #F9FAFB;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        margin-bottom: 18px;
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
    "blk": "black",
    "bk": "black",
    "wht": "white",
    "gry": "grey",
    "assy": "assembly",
    "asm": "assembly",
    "pcb": "printed circuit board",
    "pcba": "printed circuit board assembly",
    "alum": "aluminium",
    "alu": "aluminium",
    "brkt": "bracket",
    "scrw": "screw",
    "scr": "screw",
    "hex": "hexagon",
    "dia": "diameter",
    "mtr": "meter",
    "mt": "meter",
    "ctrl": "control",
    "conn": "connector",
    "pwr": "power",
    "sw": "switch",
    "swi": "switch",
    "qty": "quantity",
    "matl": "material",
    "temp": "temperature",
    "assy": "assembly",
}


def normalize_abbreviations(text):
    """
    Convert common inventory abbreviations into standard terms.
    """
    for short_form, full_form in ABBREVIATION_DICTIONARY.items():
        text = re.sub(
            rf"\b{re.escape(short_form)}\b",
            full_form,
            text,
            flags=re.IGNORECASE
        )
    return text


def clean_text(text):
    """
    Main NLP preprocessing function.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = normalize_abbreviations(text)

    # Normalize common measurement patterns
    text = re.sub(r"(\bm\d+)\s*[xX*]\s*(\d+)", r"\1 x \2", text)
    text = re.sub(r"(\d+)\s*mm\b", r"\1 mm", text)
    text = re.sub(r"(\d+)\s*cm\b", r"\1 cm", text)
    text = re.sub(r"(\d+)\s*m\b", r"\1 meter", text)
    text = re.sub(r"(\d+)\s*v\b", r"\1 volt", text)
    text = re.sub(r"(\d+)\s*a\b", r"\1 ampere", text)

    # Remove symbols
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_matched_keywords(text1, text2):
    """
    Find matched keywords between two cleaned descriptions.
    """
    words1 = set(str(text1).split())
    words2 = set(str(text2).split())

    stop_words = {
        "and", "or", "the", "for", "with", "of", "to", "in", "on",
        "a", "an", "item", "material"
    }

    matched = words1.intersection(words2)

    matched = [
        word for word in matched
        if len(word) > 1 and word not in stop_words
    ]

    return ", ".join(sorted(matched))


# ============================================================
# FILE READING
# ============================================================

def read_uploaded_file(uploaded_file):
    """
    Read CSV or Excel file.
    Supports comma, semicolon, tab, pipe separators and common encodings.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        separators = [";", ",", "\t", "|", None]
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

        best_df = None
        best_score = -1

        for encoding in encodings:
            for separator in separators:
                try:
                    uploaded_file.seek(0)

                    df = pd.read_csv(
                        uploaded_file,
                        sep=separator,
                        engine="python",
                        encoding=encoding,
                        on_bad_lines="skip"
                    )

                    if df.empty:
                        continue

                    score = len(df.columns) * 1000 + len(df)

                    if score > best_score:
                        best_score = score
                        best_df = df

                except Exception:
                    continue

        if best_df is not None:
            return best_df

        raise ValueError("Unable to read CSV file. Please check the file format.")

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


# ============================================================
# AUTO COLUMN SELECTION
# ============================================================

def auto_select_code_column(columns):
    keywords = [
        "part_code",
        "item_code",
        "material_code",
        "comp_item_id",
        "item_id",
        "code",
        "id"
    ]

    lower_map = {col.lower(): col for col in columns}

    for key in keywords:
        if key in lower_map:
            return lower_map[key]

    return columns[0]


def auto_select_description_column(columns):
    keywords = [
        "itm_desc",
        "description",
        "item_description",
        "material_description",
        "internal",
        "external",
        "part_name",
        "name"
    ]

    lower_map = {col.lower(): col for col in columns}

    for key in keywords:
        if key in lower_map:
            return lower_map[key]

    return columns[0]


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def get_result_label(score, duplicate_threshold, possible_threshold):
    if score >= duplicate_threshold:
        return "Likely Duplicate"
    elif score >= possible_threshold:
        return "Possible Duplicate"
    return "Not Duplicate"


def get_risk_level(result_label):
    if result_label == "Likely Duplicate":
        return "High"
    elif result_label == "Possible Duplicate":
        return "Medium"
    return "Low"


def get_recommendation(result_label):
    if result_label == "Likely Duplicate":
        return "High priority review. Merge or deactivate duplicate record if confirmed."
    elif result_label == "Possible Duplicate":
        return "Manual checking recommended before making changes."
    return "No action required."


def detect_duplicates(
    df,
    code_column,
    description_column,
    duplicate_threshold,
    possible_threshold,
    max_records
):
    """
    Detect duplicate inventory records using TF-IDF + Cosine Similarity.
    """
    working_df = df.copy()

    working_df[description_column] = working_df[description_column].fillna("").astype(str)
    working_df["Cleaned Description"] = working_df[description_column].apply(clean_text)

    working_df = working_df[
        working_df["Cleaned Description"].str.strip() != ""
    ].reset_index(drop=True)

    if len(working_df) > max_records:
        working_df = working_df.head(max_records).reset_index(drop=True)

    if len(working_df) < 2:
        return working_df, pd.DataFrame(), pd.DataFrame()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(working_df["Cleaned Description"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    records = []

    for i in range(len(working_df)):
        for j in range(i + 1, len(working_df)):
            score = float(similarity_matrix[i][j])

            result_label = get_result_label(
                score,
                duplicate_threshold,
                possible_threshold
            )

            if result_label == "Not Duplicate":
                continue

            item_code_1 = working_df.loc[i, code_column]
            item_code_2 = working_df.loc[j, code_column]

            description_1 = working_df.loc[i, description_column]
            description_2 = working_df.loc[j, description_column]

            cleaned_1 = working_df.loc[i, "Cleaned Description"]
            cleaned_2 = working_df.loc[j, "Cleaned Description"]

            records.append({
                "Item Code 1": item_code_1,
                "Item Description 1": description_1,
                "Item Code 2": item_code_2,
                "Item Description 2": description_2,
                "Similarity (%)": round(score * 100, 2),
                "Duplicate Status": result_label,
                "Risk Level": get_risk_level(result_label),
                "Recommended Action": get_recommendation(result_label),
                "Matched Keywords": get_matched_keywords(cleaned_1, cleaned_2),
                "Cleaned Description 1": cleaned_1,
                "Cleaned Description 2": cleaned_2,
            })

    result_df = pd.DataFrame(records)

    if result_df.empty:
        return working_df, result_df, result_df

    # Remove repeated pair based on item code combination
    result_df["Pair Key"] = result_df.apply(
        lambda row: tuple(sorted([
            str(row["Item Code 1"]),
            str(row["Item Code 2"])
        ])),
        axis=1
    )

    result_df = result_df.sort_values(
        by="Similarity (%)",
        ascending=False
    )

    result_df = result_df.drop_duplicates(
        subset=["Pair Key"],
        keep="first"
    ).drop(columns=["Pair Key"]).reset_index(drop=True)

    user_result_df = result_df[
        [
            "Item Code 1",
            "Item Description 1",
            "Item Code 2",
            "Item Description 2",
            "Similarity (%)",
            "Duplicate Status",
            "Risk Level",
            "Recommended Action",
            "Matched Keywords"
        ]
    ]

    technical_result_df = result_df[
        [
            "Item Code 1",
            "Item Code 2",
            "Similarity (%)",
            "Duplicate Status",
            "Cleaned Description 1",
            "Cleaned Description 2"
        ]
    ]

    return working_df, user_result_df, technical_result_df


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def convert_actual_label(label):
    label = str(label).strip().lower()

    if label in ["duplicate", "yes", "1", "true", "same"]:
        return 1

    return 0


def convert_predicted_label(score, duplicate_threshold, possible_threshold):
    """
    For evaluation:
    Likely Duplicate and Possible Duplicate are treated as Duplicate.
    """
    if score >= possible_threshold:
        return 1

    return 0


def evaluate_labelled_pairs(eval_df, desc1_col, desc2_col, label_col, duplicate_threshold, possible_threshold):
    """
    Evaluate using labelled item-description pairs.
    """
    eval_df = eval_df.copy()

    eval_df[desc1_col] = eval_df[desc1_col].fillna("").astype(str)
    eval_df[desc2_col] = eval_df[desc2_col].fillna("").astype(str)

    eval_df["cleaned_1"] = eval_df[desc1_col].apply(clean_text)
    eval_df["cleaned_2"] = eval_df[desc2_col].apply(clean_text)

    all_descriptions = pd.concat([
        eval_df["cleaned_1"],
        eval_df["cleaned_2"]
    ], ignore_index=True)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2)
    )

    vectorizer.fit(all_descriptions)

    vec1 = vectorizer.transform(eval_df["cleaned_1"])
    vec2 = vectorizer.transform(eval_df["cleaned_2"])

    scores = []

    for i in range(len(eval_df)):
        score = cosine_similarity(vec1[i], vec2[i])[0][0]
        scores.append(float(score))

    eval_df["Similarity (%)"] = [round(score * 100, 2) for score in scores]
    eval_df["Actual Binary"] = eval_df[label_col].apply(convert_actual_label)
    eval_df["Predicted Binary"] = [
        convert_predicted_label(score, duplicate_threshold, possible_threshold)
        for score in scores
    ]

    eval_df["Predicted Label"] = eval_df["Predicted Binary"].apply(
        lambda x: "Duplicate" if x == 1 else "Not Duplicate"
    )

    accuracy = accuracy_score(eval_df["Actual Binary"], eval_df["Predicted Binary"])
    precision = precision_score(eval_df["Actual Binary"], eval_df["Predicted Binary"], zero_division=0)
    recall = recall_score(eval_df["Actual Binary"], eval_df["Predicted Binary"], zero_division=0)
    f1 = f1_score(eval_df["Actual Binary"], eval_df["Predicted Binary"], zero_division=0)

    cm = confusion_matrix(eval_df["Actual Binary"], eval_df["Predicted Binary"])

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Confusion Matrix": cm
    }

    output_df = eval_df[
        [
            desc1_col,
            desc2_col,
            label_col,
            "Similarity (%)",
            "Predicted Label"
        ]
    ]

    return metrics, output_df


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def dataframe_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel(df, sheet_name="Results"):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    return output.getvalue()


# ============================================================
# SAMPLE DATA
# ============================================================

def create_sample_data():
    return pd.DataFrame({
        "ITEM_CODE": [
            "MAT001", "MAT002", "MAT003", "MAT004",
            "MAT005", "MAT006", "MAT007", "MAT008"
        ],
        "ITM_DESC": [
            "Screw M4 x 10mm stainless steel",
            "SS screw M4 10mm",
            "Rubber gasket black 20mm",
            "Black rubber seal 20 mm",
            "PCB controller board",
            "Main control PCB board",
            "Cable black 2 meter",
            "Cable blk 2m"
        ]
    })


def create_sample_evaluation_data():
    return pd.DataFrame({
        "description_1": [
            "SS screw M4 10mm",
            "Cable blk 2m",
            "PCB controller board",
            "Rubber gasket black 20mm",
            "Screw M4 x 10mm stainless steel",
            "Cable black 2 meter",
            "Steel bracket L shape",
            "Plastic cover white"
        ],
        "description_2": [
            "Screw M4 x 10mm stainless steel",
            "Cable black 2 meter",
            "Main control PCB board",
            "Black rubber seal 20 mm",
            "PCB controller board",
            "Rubber gasket black 20mm",
            "Temperature sensor module",
            "Hex bolt M6 x 30mm"
        ],
        "actual_label": [
            "Duplicate",
            "Duplicate",
            "Duplicate",
            "Duplicate",
            "Not Duplicate",
            "Not Duplicate",
            "Not Duplicate",
            "Not Duplicate"
        ]
    })


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Detection Settings")

duplicate_threshold = st.sidebar.slider(
    "Likely duplicate threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.80,
    step=0.01
)

possible_threshold = st.sidebar.slider(
    "Possible duplicate threshold",
    min_value=0.10,
    max_value=0.79,
    value=0.50,
    step=0.01
)

max_records = st.sidebar.slider(
    "Maximum records to process",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100,
    help="Higher values may take longer because every row is compared with other rows."
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### NLP Components

    **Text Preprocessing**  
    Cleans item descriptions.

    **Abbreviation Normalization**  
    Converts short forms such as SS, PCB, BLK.

    **TF-IDF Vectorization**  
    Converts text into numerical features.

    **Cosine Similarity**  
    Measures similarity between item descriptions.
    """
)

if possible_threshold >= duplicate_threshold:
    st.sidebar.error("Possible threshold must be lower than likely duplicate threshold.")


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown(
    '<div class="main-title">📦 Inventory Duplicate Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP-based duplicate detection for inventory and material master item descriptions.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    <b>Objective:</b> This application identifies inventory records that may refer to the same item,
    even when their descriptions are written differently. The system compares item descriptions using
    NLP techniques and highlights records that require manual review.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

tab_detection, tab_evaluation = st.tabs(
    ["🔍 Duplicate Detection", "📊 Model Evaluation"]
)


# ============================================================
# TAB 1: DUPLICATE DETECTION
# ============================================================

with tab_detection:

    st.markdown(
        '<div class="section-title">1. Upload Dataset</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload inventory or material master file",
        type=["csv", "xlsx", "xls"],
        key="main_dataset"
    )

    use_sample_data = st.checkbox("Use sample dataset")

    df = None
    current_source_key = None

    if use_sample_data:
        df = create_sample_data()
        current_source_key = "sample_dataset"
        st.success("Sample dataset loaded successfully.")

    elif uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)
            current_source_key = f"{uploaded_file.name}_{uploaded_file.size}"
            st.success("File uploaded successfully.")
        except Exception as error:
            st.error(f"Failed to read file: {error}")

    if current_source_key != st.session_state.source_key:
        reset_detection_state()
        st.session_state.source_key = current_source_key

    if df is not None:

        st.markdown(
            '<div class="section-title">2. Dataset Overview</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Method", "TF-IDF + Cosine Similarity")

        if len(df) > max_records:
            st.warning(
                f"Your dataset has {len(df)} records. Only the first {max_records} records "
                f"will be processed based on the current sidebar setting."
            )

        st.dataframe(
            df.head(30),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="section-title">3. Select Required Columns</div>',
            unsafe_allow_html=True
        )

        columns = df.columns.tolist()

        default_code = auto_select_code_column(columns)
        default_desc = auto_select_description_column(columns)

        col_a, col_b = st.columns(2)

        with col_a:
            code_column = st.selectbox(
                "Item code column",
                options=columns,
                index=columns.index(default_code)
            )

        with col_b:
            description_column = st.selectbox(
                "Item description column",
                options=columns,
                index=columns.index(default_desc)
            )

        st.markdown(
            """
            <div class="method-box">
            <b>Detection logic:</b> The system compares the selected item description from one row
            with the selected item description from another row. The item code is only used to identify
            the records in the result.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">4. Run Duplicate Detection</div>',
            unsafe_allow_html=True
        )

        run_button = st.button(
            "🚀 Run Duplicate Detection",
            type="primary",
            use_container_width=True
        )

        if run_button:

            if possible_threshold >= duplicate_threshold:
                st.error("Possible duplicate threshold must be lower than likely duplicate threshold.")

            else:
                with st.spinner("Processing descriptions and calculating similarity..."):
                    cleaned_df, result_df, technical_result_df = detect_duplicates(
                        df=df,
                        code_column=code_column,
                        description_column=description_column,
                        duplicate_threshold=duplicate_threshold,
                        possible_threshold=possible_threshold,
                        max_records=max_records
                    )

                st.session_state.cleaned_df = cleaned_df
                st.session_state.result_df = result_df
                st.session_state.technical_result_df = technical_result_df
                st.session_state.detection_done = True
                st.session_state.selected_code_column = code_column
                st.session_state.selected_description_column = description_column

        if st.session_state.detection_done:

            cleaned_df = st.session_state.cleaned_df
            result_df = st.session_state.result_df
            technical_result_df = st.session_state.technical_result_df

            saved_code_column = st.session_state.selected_code_column
            saved_description_column = st.session_state.selected_description_column

            st.markdown(
                '<div class="section-title">5. Detection Summary</div>',
                unsafe_allow_html=True
            )

            if result_df.empty:
                st.warning("No duplicate or possible duplicate records were found. Try lowering the threshold.")

            else:
                high_risk = len(result_df[result_df["Risk Level"] == "High"])
                medium_risk = len(result_df[result_df["Risk Level"] == "Medium"])

                s1, s2, s3 = st.columns(3)
                s1.metric("High Risk Matches", high_risk)
                s2.metric("Medium Risk Matches", medium_risk)
                s3.metric("Total Matches Found", len(result_df))

                summary_df = result_df["Duplicate Status"].value_counts().reset_index()
                summary_df.columns = ["Duplicate Status", "Count"]

                fig, ax = plt.subplots()
                ax.bar(summary_df["Duplicate Status"], summary_df["Count"])
                ax.set_xlabel("Duplicate Status")
                ax.set_ylabel("Number of Matches")
                ax.set_title("Duplicate Detection Summary")
                plt.xticks(rotation=15)

                st.pyplot(fig)

                st.markdown(
                    '<div class="section-title">6. Review Potential Duplicate Records</div>',
                    unsafe_allow_html=True
                )

                st.info(
                    "Review the records below. Start with High Risk matches because they are the most likely duplicates."
                )

                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    status_filter = st.selectbox(
                        "Filter by duplicate status",
                        options=["All"] + sorted(result_df["Duplicate Status"].unique().tolist()),
                        key="status_filter"
                    )

                with filter_col2:
                    risk_filter = st.selectbox(
                        "Filter by risk level",
                        options=["All"] + sorted(result_df["Risk Level"].unique().tolist()),
                        key="risk_filter"
                    )

                display_df = result_df.copy()

                if status_filter != "All":
                    display_df = display_df[
                        display_df["Duplicate Status"] == status_filter
                    ]

                if risk_filter != "All":
                    display_df = display_df[
                        display_df["Risk Level"] == risk_filter
                    ]

                if len(display_df) == 0:
                    st.warning("No records match the selected filter.")

                else:
                    max_display = min(500, len(display_df))

                    if max_display < 10:
                        display_limit = max_display
                    else:
                        display_limit = st.slider(
                            "Number of records to display",
                            min_value=10,
                            max_value=max_display,
                            value=min(50, max_display),
                            step=10,
                            key="display_limit"
                        )

                    st.dataframe(
                        display_df.head(display_limit),
                        use_container_width=True,
                        hide_index=True
                    )

                with st.expander("View cleaned text used by NLP model"):
                    st.write(
                        "This table shows the cleaned descriptions after preprocessing. "
                        "It is useful for explaining the NLP process in your report."
                    )

                    if (
                        saved_code_column in cleaned_df.columns
                        and saved_description_column in cleaned_df.columns
                        and "Cleaned Description" in cleaned_df.columns
                    ):
                        st.dataframe(
                            cleaned_df[
                                [saved_code_column, saved_description_column, "Cleaned Description"]
                            ].head(100),
                            use_container_width=True,
                            hide_index=True
                        )

                with st.expander("View technical similarity results"):
                    st.dataframe(
                        technical_result_df.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                st.markdown(
                    '<div class="section-title">7. Download Results</div>',
                    unsafe_allow_html=True
                )

                d1, d2 = st.columns(2)

                with d1:
                    st.download_button(
                        label="📥 Download Review Results CSV",
                        data=dataframe_to_csv(result_df),
                        file_name="inventory_duplicate_review_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with d2:
                    st.download_button(
                        label="📥 Download Review Results Excel",
                        data=dataframe_to_excel(result_df),
                        file_name="inventory_duplicate_review_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                st.markdown(
                    '<div class="section-title">8. NLP Method Explanation</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    This application uses a text similarity approach to detect potential duplicate inventory records.

                    **Process flow:**

                    1. **Text preprocessing** removes symbols, converts text to lowercase, and cleans spacing.
                    2. **Abbreviation normalization** converts common short forms into standard words.
                    3. **TF-IDF vectorization** transforms item descriptions into numerical text features.
                    4. **Cosine similarity** compares one item description with another item description.
                    5. **Threshold classification** labels the pair as Likely Duplicate or Possible Duplicate.

                    The item code is not used to calculate similarity. It is only used as a reference to identify the records.
                    """
                )

    else:
        st.info("Upload a CSV/Excel file or use the sample dataset to begin.")

        st.markdown(
            '<div class="section-title">Sample Dataset Format</div>',
            unsafe_allow_html=True
        )

        sample_df = create_sample_data()

        st.dataframe(
            sample_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="📥 Download Sample Dataset",
            data=dataframe_to_csv(sample_df),
            file_name="sample_inventory_dataset.csv",
            mime="text/csv"
        )


# ============================================================
# TAB 2: MODEL EVALUATION
# ============================================================

with tab_evaluation:

    st.markdown(
        '<div class="section-title">Model Evaluation Using Labelled Pairs</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Use this section for your report. Upload a labelled dataset with two descriptions and the actual label."
    )

    eval_uploaded_file = st.file_uploader(
        "Upload labelled evaluation file",
        type=["csv", "xlsx", "xls"],
        key="eval_dataset"
    )

    use_sample_eval = st.checkbox("Use sample evaluation dataset")

    eval_df = None

    if use_sample_eval:
        eval_df = create_sample_evaluation_data()
        st.success("Sample evaluation dataset loaded successfully.")

    elif eval_uploaded_file is not None:
        try:
            eval_df = read_uploaded_file(eval_uploaded_file)
            st.success("Evaluation file uploaded successfully.")
        except Exception as error:
            st.error(f"Failed to read evaluation file: {error}")

    if eval_df is not None:

        st.markdown("### Evaluation Dataset Preview")
        st.dataframe(eval_df.head(30), use_container_width=True, hide_index=True)

        eval_columns = eval_df.columns.tolist()

        default_desc1 = "description_1" if "description_1" in eval_columns else eval_columns[0]
        default_desc2 = "description_2" if "description_2" in eval_columns else eval_columns[min(1, len(eval_columns) - 1)]
        default_label = "actual_label" if "actual_label" in eval_columns else eval_columns[-1]

        e1, e2, e3 = st.columns(3)

        with e1:
            desc1_col = st.selectbox(
                "First description column",
                options=eval_columns,
                index=eval_columns.index(default_desc1),
                key="desc1_col"
            )

        with e2:
            desc2_col = st.selectbox(
                "Second description column",
                options=eval_columns,
                index=eval_columns.index(default_desc2),
                key="desc2_col"
            )

        with e3:
            label_col = st.selectbox(
                "Actual label column",
                options=eval_columns,
                index=eval_columns.index(default_label),
                key="label_col"
            )

        if st.button("📊 Run Evaluation", type="primary", use_container_width=True):

            try:
                metrics, evaluation_result_df = evaluate_labelled_pairs(
                    eval_df=eval_df,
                    desc1_col=desc1_col,
                    desc2_col=desc2_col,
                    label_col=label_col,
                    duplicate_threshold=duplicate_threshold,
                    possible_threshold=possible_threshold
                )

                st.markdown("### Evaluation Metrics")

                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
                m2.metric("Precision", f"{metrics['Precision']:.2f}")
                m3.metric("Recall", f"{metrics['Recall']:.2f}")
                m4.metric("F1-score", f"{metrics['F1-score']:.2f}")

                st.markdown("### Confusion Matrix")

                cm = metrics["Confusion Matrix"]

                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Not Duplicate", "Actual Duplicate"],
                    columns=["Predicted Not Duplicate", "Predicted Duplicate"]
                )

                st.dataframe(cm_df, use_container_width=True)

                st.markdown("### Evaluation Results")
                st.dataframe(evaluation_result_df, use_container_width=True, hide_index=True)

                st.download_button(
                    label="📥 Download Evaluation Results CSV",
                    data=dataframe_to_csv(evaluation_result_df),
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as error:
                st.error(f"Evaluation failed: {error}")

    else:
        st.markdown("### Sample Evaluation Dataset Format")

        sample_eval_df = create_sample_evaluation_data()

        st.dataframe(
            sample_eval_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="📥 Download Sample Evaluation Dataset",
            data=dataframe_to_csv(sample_eval_df),
            file_name="sample_evaluation_dataset.csv",
            mime="text/csv"
        )
