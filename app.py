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
    layout="wide"
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
    "cord": "cord",
    "sw": "switch",
    "swi": "switch",
}


def normalize_abbreviations(text):
    """
    Replace common inventory/material abbreviations with full terms.
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
    Clean inventory item description before NLP processing.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = normalize_abbreviations(text)

    # Normalize measurements and patterns
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


def get_matched_words(text1, text2):
    """
    Find common words between two cleaned descriptions.
    """
    words1 = set(str(text1).split())
    words2 = set(str(text2).split())

    matched = words1.intersection(words2)

    # Remove very short words
    matched = [word for word in matched if len(word) > 1]

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
                    pass

        if best_df is not None:
            return best_df

        raise ValueError("Unable to read CSV file.")

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type.")


# ============================================================
# AUTO COLUMN SELECTION
# ============================================================

def auto_select_code_column(columns):
    """
    Auto-select likely item code column.
    """
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
    """
    Auto-select likely item description column.
    """
    keywords = [
        "itm_desc",
        "description",
        "item_description",
        "material_description",
        "internal",
        "external",
        "part_name"
    ]

    lower_map = {col.lower(): col for col in columns}

    for key in keywords:
        if key in lower_map:
            return lower_map[key]

    return columns[0]


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def get_prediction(score, duplicate_threshold, possible_threshold):
    """
    Convert similarity score into duplicate label.
    """
    if score >= duplicate_threshold:
        return "Duplicate"
    elif score >= possible_threshold:
        return "Possible Duplicate"
    else:
        return "Not Duplicate"


def get_recommendation(prediction):
    """
    Convert technical prediction into simple user action.
    """
    if prediction == "Duplicate":
        return "Likely same item - review and merge if confirmed"
    elif prediction == "Possible Duplicate":
        return "Similar item - manual checking needed"
    else:
        return "Different item"


def detect_duplicates(
    df,
    code_column,
    description_column,
    duplicate_threshold,
    possible_threshold
):
    """
    Detect duplicate inventory items by comparing one row description
    with another row description.

    Item code is used only as identifier, not for similarity calculation.
    """
    working_df = df.copy()

    working_df[description_column] = working_df[description_column].fillna("").astype(str)
    working_df["cleaned_description"] = working_df[description_column].apply(clean_text)

    working_df = working_df[
        working_df["cleaned_description"].str.strip() != ""
    ].reset_index(drop=True)

    if len(working_df) < 2:
        return working_df, pd.DataFrame(), pd.DataFrame()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(working_df["cleaned_description"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results = []

    for i in range(len(working_df)):
        for j in range(i + 1, len(working_df)):
            score = similarity_matrix[i][j]

            prediction = get_prediction(
                score,
                duplicate_threshold,
                possible_threshold
            )

            if prediction == "Not Duplicate":
                continue

            item_code_1 = working_df.loc[i, code_column]
            item_code_2 = working_df.loc[j, code_column]

            desc_1 = working_df.loc[i, description_column]
            desc_2 = working_df.loc[j, description_column]

            clean_1 = working_df.loc[i, "cleaned_description"]
            clean_2 = working_df.loc[j, "cleaned_description"]

            row = {
                "Item Code 1": item_code_1,
                "Item Description 1": desc_1,
                "Item Code 2": item_code_2,
                "Item Description 2": desc_2,
                "Similarity (%)": round(score * 100, 2),
                "Result": prediction,
                "Recommendation": get_recommendation(prediction),
                "Matched Keywords": get_matched_words(clean_1, clean_2),
                "Cleaned Description 1": clean_1,
                "Cleaned Description 2": clean_2,
            }

            results.append(row)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return working_df, result_df, result_df

    # Remove repeated same item-code pair
    result_df["pair_key"] = result_df.apply(
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
        subset=["pair_key"],
        keep="first"
    )

    result_df = result_df.drop(columns=["pair_key"]).reset_index(drop=True)

    simple_result_df = result_df[
        [
            "Item Code 1",
            "Item Description 1",
            "Item Code 2",
            "Item Description 2",
            "Similarity (%)",
            "Result",
            "Recommendation",
            "Matched Keywords"
        ]
    ]

    technical_result_df = result_df[
        [
            "Item Code 1",
            "Item Code 2",
            "Similarity (%)",
            "Result",
            "Cleaned Description 1",
            "Cleaned Description 2"
        ]
    ]

    return working_df, simple_result_df, technical_result_df


# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================

def dataframe_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")

    return output.getvalue()


# ============================================================
# SAMPLE DATA
# ============================================================

def create_sample_data():
    return pd.DataFrame({
        "ITEM_CODE": [
            "MAT001",
            "MAT002",
            "MAT003",
            "MAT004",
            "MAT005",
            "MAT006",
            "MAT007",
            "MAT008"
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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

duplicate_threshold = st.sidebar.slider(
    "Duplicate threshold",
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

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### NLP Method

    - Text preprocessing
    - Abbreviation normalization
    - TF-IDF vectorization
    - Cosine similarity
    - Threshold classification
    """
)


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown(
    '<div class="main-title">📦 Duplicate Inventory Item Detection Using NLP</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect duplicate or similar inventory item descriptions using TF-IDF and Cosine Similarity.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    This system compares one inventory item description with another item description.
    The item code is used only as an identifier in the result table.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD DATA
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Inventory Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

use_sample_data = st.checkbox("Use sample dataset")

df = None

if use_sample_data:
    df = create_sample_data()
    st.success("Sample dataset loaded successfully.")

elif uploaded_file is not None:
    try:
        df = read_uploaded_file(uploaded_file)
        st.success("File uploaded successfully.")
    except Exception as error:
        st.error(f"Failed to read file: {error}")


# ============================================================
# PROCESS DATA
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

    columns = df.columns.tolist()

    default_code = auto_select_code_column(columns)
    default_desc = auto_select_description_column(columns)

    code_column = st.selectbox(
        "Select item code column",
        options=columns,
        index=columns.index(default_code)
    )

    description_column = st.selectbox(
        "Select item description column",
        options=columns,
        index=columns.index(default_desc)
    )

    st.markdown(
        '<div class="section-title">4. Run Detection</div>',
        unsafe_allow_html=True
    )

    if st.button("🚀 Detect Duplicate Items", type="primary"):

        if possible_threshold >= duplicate_threshold:
            st.error("Possible duplicate threshold must be lower than duplicate threshold.")

        else:
            with st.spinner("Running NLP duplicate detection..."):
                cleaned_df, result_df, technical_result_df = detect_duplicates(
                    df=df,
                    code_column=code_column,
                    description_column=description_column,
                    duplicate_threshold=duplicate_threshold,
                    possible_threshold=possible_threshold
                )

            st.markdown(
                '<div class="section-title">5. Cleaned Text Preview</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                cleaned_df[[code_column, description_column, "cleaned_description"]].head(50),
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                '<div class="section-title">6. Detection Summary</div>',
                unsafe_allow_html=True
            )

            if result_df.empty:
                st.warning("No duplicate or possible duplicate items detected. Try lowering the threshold.")

            else:
                duplicate_count = len(result_df[result_df["Result"] == "Duplicate"])
                possible_count = len(result_df[result_df["Result"] == "Possible Duplicate"])

                m1, m2, m3 = st.columns(3)
                m1.metric("Duplicate", duplicate_count)
                m2.metric("Possible Duplicate", possible_count)
                m3.metric("Total Similar Pairs", len(result_df))

                summary_df = result_df["Result"].value_counts().reset_index()
                summary_df.columns = ["Result", "Count"]

                fig, ax = plt.subplots()
                ax.bar(summary_df["Result"], summary_df["Count"])
                ax.set_xlabel("Result")
                ax.set_ylabel("Count")
                ax.set_title("Duplicate Detection Summary")
                st.pyplot(fig)

                st.markdown(
                    '<div class="section-title">7. Detection Results</div>',
                    unsafe_allow_html=True
                )

                st.info(
                    "The table below shows item pairs that may be duplicates. "
                    "Focus on the Similarity (%), Result, and Recommendation columns."
                )

                result_filter = st.selectbox(
                    "Show result type",
                    options=["All", "Duplicate", "Possible Duplicate"]
                )

                display_df = result_df.copy()

                if result_filter != "All":
                    display_df = display_df[display_df["Result"] == result_filter]

                if len(display_df) > 0:
                    top_limit = st.slider(
                        "Number of results to display",
                        min_value=10,
                        max_value=max(10, min(500, len(display_df))),
                        value=min(50, len(display_df)),
                        step=10
                    )

                    st.dataframe(
                        display_df.head(top_limit),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No results for the selected filter.")

                with st.expander("View technical NLP details"):
                    st.write(
                        "This section shows the cleaned text used by the NLP model. "
                        "You can use this for your report or presentation explanation."
                    )
                    st.dataframe(
                        technical_result_df.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                st.markdown(
                    '<div class="section-title">8. Download Results</div>',
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.download_button(
                        label="📥 Download Simple Results CSV",
                        data=dataframe_to_csv(result_df),
                        file_name="inventory_duplicate_simple_results.csv",
                        mime="text/csv"
                    )

                with c2:
                    st.download_button(
                        label="📥 Download Simple Results Excel",
                        data=dataframe_to_excel(result_df),
                        file_name="inventory_duplicate_simple_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                with st.expander("Download technical NLP results"):
                    c3, c4 = st.columns(2)

                    with c3:
                        st.download_button(
                            label="📥 Download Technical CSV",
                            data=dataframe_to_csv(technical_result_df),
                            file_name="inventory_duplicate_technical_results.csv",
                            mime="text/csv"
                        )

                    with c4:
                        st.download_button(
                            label="📥 Download Technical Excel",
                            data=dataframe_to_excel(technical_result_df),
                            file_name="inventory_duplicate_technical_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                st.markdown(
                    '<div class="section-title">9. NLP Method Explanation</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    This system uses NLP to compare inventory item descriptions.

                    **How it works:**

                    1. The item description is cleaned.
                    2. Common abbreviations are standardized.
                    3. The cleaned text is converted into TF-IDF vectors.
                    4. Cosine similarity compares one row with another row.
                    5. The system labels the result as Duplicate or Possible Duplicate.

                    **Important:** The item code is not used to calculate similarity. It is only used to show which two items may be duplicates.
                    """
                )

else:
    st.info("Please upload a CSV/Excel file or use the sample dataset.")

    st.markdown(
        '<div class="section-title">Sample Dataset Format</div>',
        unsafe_allow_html=True
    )

    sample_df = create_sample_data()
    st.dataframe(sample_df, use_container_width=True)

    st.download_button(
        label="📥 Download Sample CSV",
        data=dataframe_to_csv(sample_df),
        file_name="sample_inventory_dataset.csv",
        mime="text/csv"
    )
