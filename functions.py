import pandas as pd

def clean_and_convert_column(series, target_type):
    if target_type == "int":
        return pd.to_numeric(series.replace(r"[^\d\-]", "", regex=True), errors="coerce").astype("Int64")
    elif target_type == "float":
        # First convert to string to handle any non-string inputs
        series_str = series.astype(str)
        # Replace commas with dots for decimal points
        series_str = series_str.str.replace(",", ".", regex=False)
        # Remove any non-numeric characters except decimal points and minus signs
        series_str = series_str.replace(r"[^\d\.\-]", "", regex=True)
        # Convert to float, coercing errors to NaN
        return pd.to_numeric(series_str, errors="coerce")
    elif target_type in ["date", "datetime"]:
        # Try multiple datetime formats
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    elif target_type == "str":
        return series.astype(str)
    else:
        return series