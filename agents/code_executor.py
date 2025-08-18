import pandas as pd

def safe_execute(code: str, df: pd.DataFrame):
    local_vars = {"df": df, "pd": pd}
    try:
        result = eval(code, {"__builtins__": {}}, local_vars)
        if isinstance(result, pd.DataFrame):
            return result.to_dict(orient="records")
        elif isinstance(result, pd.Series):
            return result.to_dict()
        else:
            return str(result)
    except Exception as e:
        return f"Execution Error: {str(e)}"
