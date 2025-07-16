import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from io import BytesIO
from datetime import datetime
import time


st.title("Product Similarity Identifier")

st.info("Please upload file with only **2 COLUMNS**: one for Item Key (NANKEY) and one for Description (PROD_DESC).")

uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx only)",
    type=["xlsx"]
)


def guess_columns(df):
    possible_nankey = None
    possible_desc = None
    for col in df.columns:
        series = df[col].astype(str)
        numeric_ratio = pd.to_numeric(series, errors='coerce').notna().mean()
        avg_length = series.astype(str).str.len().mean()

        if numeric_ratio > 0.7 and possible_nankey is None:
            possible_nankey = col
        elif avg_length > 5 and possible_desc is None:
            possible_desc = col

    return possible_nankey, possible_desc


if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, header=0)
        df.columns = df.columns.astype(str).str.strip().str.upper()

        if df.shape[1] > 2:
            st.warning("File has more than 2 columns. Please upload only NANKEY and PROD_DESC columns ONLY.")
            st.stop()

        nankey_col, desc_col = guess_columns(df)

        if not nankey_col or not desc_col:
            st.warning("Unable to detect NANKEY and PROD_DESC. Please select manually.")
            nankey_col = st.selectbox("Select the column for NANKEY (typically numbers)", df.columns)
            desc_col = st.selectbox("Select the column for PROD_DESC (typically text)", df.columns)
        else:
            st.info(f"Auto-detected columns: NANKEY → **{nankey_col}**, PROD_DESC → **{desc_col}**")

        df = df[[nankey_col, desc_col]]
        df.columns = ["NANKEY", "PROD_DESC"]

        threshold = st.slider("Select Similarity Threshold (0-100)", min_value=0, max_value=100, value=60, step=1)

        if st.button("Process"):
            df["PROD_DESC"] = df["PROD_DESC"].astype(str).str.lower().str.strip()
            descriptions = df["PROD_DESC"].tolist()
            n = len(descriptions)
            similarity_matrix = np.zeros((n, n))
            start_time = time.time()

            for i in range(n):
                similarities = [fuzz.token_sort_ratio(descriptions[i], descriptions[j]) for j in range(i + 1, n)]
                similarity_matrix[i, i + 1:] = similarities
                similarity_matrix[i + 1:, i] = similarities

            used_indices = set()
            bulk_groups = []
            unique_items = []

            for i in range(n):
                if i in used_indices:
                    continue
                group = [i]
                for j in range(i + 1, n):
                    if j in used_indices:
                        continue
                    if similarity_matrix[i, j] >= threshold:
                        group.append(j)
                        used_indices.add(j)
                if len(group) > 1:
                    bulk_groups.append(group)
                    used_indices.update(group)
                else:
                    unique_items.append(i)

            bulk_data = []
            bulk_group_id = 1
            for group in bulk_groups:
                for idx in group:
                    nankey = df.loc[idx, 'NANKEY']
                    description = df.loc[idx, 'PROD_DESC']
                    bulk_data.append([bulk_group_id, nankey, description])
                bulk_group_id += 1

            unique_data = []
            for idx in unique_items:
                nankey = df.loc[idx, 'NANKEY']
                description = df.loc[idx, 'PROD_DESC']
                unique_data.append([nankey, description])

            bulk_df = pd.DataFrame(bulk_data, columns=['Bulk Group ID', 'NANKEY', 'PROD_DESC'])
            unique_df = pd.DataFrame(unique_data, columns=['NANKEY', 'PROD_DESC'])

            pivot_df = bulk_df.groupby('Bulk Group ID').size().reset_index(name='Count of Bulk Group ID')
            pivot_df = pivot_df.sort_values(by='Count of Bulk Group ID', ascending=False)

            output = BytesIO()
            today_str = datetime.today().strftime('%d%m%y')
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                unique_df.to_excel(writer, sheet_name='Unique Items', index=False)
                bulk_df.to_excel(writer, sheet_name='Bulk Items', index=False)
                pivot_df.to_excel(writer, sheet_name='Pivots', index=False)

            st.success("Download your output below.")
            st.download_button(
                label="📥 Download Excel Output",
                data=output.getvalue(),
                file_name=f"bulk_identifier_output_{today_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            end_time = time.time()
            st.write(f"Execution Time: {end_time - start_time:.2f} seconds")
            st.write(f"Unique Items: {len(unique_items)}")
            st.write(f"Bulk Groups: {len(bulk_groups)}")
            st.write(f"Bulk Items: {sum(len(group) for group in bulk_groups)}")
            st.write(f"Total Items Processed: {n}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
