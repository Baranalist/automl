import pandas as pd
import numpy as np
from datetime import datetime

def clean_and_convert_column(series, target_type):
    """Clean and convert a pandas Series to the specified type"""
    if target_type == "int":
        return pd.to_numeric(series.replace(r"[^\d\-]", "", regex=True), errors="coerce").astype("Int64")
    elif target_type == "float":
        return pd.to_numeric(series.replace(r"[^\d\.\-]", "", regex=True).str.replace(",", ".", regex=False), errors="coerce")
    elif target_type in ["date", "datetime"]:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    elif target_type == "str":
        return series.astype(str)
    else:
        return series

def get_column_metadata(df, col_types):
    """Get metadata for all columns in the dataframe"""
    meta_info = []
    for col in df.columns:
        inferred_type = pd.api.types.infer_dtype(df[col], skipna=True)
        meta_info.append({
            "column": col,
            "inferred_type": inferred_type,
            "selected_type": col_types[col],
            "unique_values": len(df[col].dropna().unique()),
            "missing_values": df[col].isna().sum()
        })
    return meta_info

def update_categorical_metadata(meta_info, col, cat_type, order=None):
    """Update metadata for categorical columns"""
    for meta in meta_info:
        if meta["column"] == col:
            meta["categorical_type"] = cat_type
            meta["ordinal_order"] = order
    return meta_info 