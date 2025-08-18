import os
import pandas as pd
from openai import OpenAI
import traceback

# Initialize OpenAI client only if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


def ask_csv_question(question: str, df: pd.DataFrame):
    """
    Takes a natural language question and a Pandas DataFrame,
    asks GPT to generate Python code to answer it,
    executes that code, and returns the result + generated code.
    """

    if client is None:
        raise RuntimeError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

    # Convert dataframe head to help GPT understand structure
    preview = df.head(5).to_dict(orient="records")
    schema = list(df.columns)

    prompt = f"""
    You are a data assistant. A user uploaded a CSV file with the following columns:
    {schema}

    The first 5 rows of data are:
    {preview}

    The user asked:
    "{question}"

    Write valid Python Pandas code that answers the question.
    - Assume the dataframe is called df
    - Store the final output in a variable named result
    - result can be a string, number, or DataFrame (if DataFrame, return head only)
    - Do NOT print anything
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    # Extract code
    code = response.choices[0].message.content
    code = code.strip().strip("```python").strip("```").strip()

    try:
        # Execute safely
        local_vars = {"df": df}
        exec(code, {}, local_vars)

        # Look for result
        result = local_vars.get("result", None)

        # Convert DataFrame to preview
        if isinstance(result, pd.DataFrame):
            result = result.head(10).to_dict(orient="records")

        return result, code

    except Exception as e:
        return f"⚠️ Error executing generated code:\n{traceback.format_exc()}", code
