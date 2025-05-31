import pandas as pd

def clean_and_convert_column(series, target_type):
    if target_type == "int":
        return pd.to_numeric(series.replace(r"[^\d\-]", "", regex=True), errors="coerce").astype("Int64")
    elif target_type == "float":
        return pd.to_numeric(series.replace(r"[^\d\.\-]", "", regex=True).str.replace(",", ".", regex=False), errors="coerce")
    elif target_type in ["date", "datetime"]:
        # Try multiple datetime formats
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    elif target_type == "str":
        return series.astype(str)
    else:
        return series