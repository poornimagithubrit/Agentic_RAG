import os
import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS for frontend JS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CSV_FILES = {}

# Serve the frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("templates", "index.html")  # place HTML here
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"detail": "index.html not found"}, status_code=404)

# Upload CSV
@app.post("/upload_csv")
async def upload_csv(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    df = pd.read_csv(file_path)
    CSV_FILES[file.filename] = df
    return {
        "message": f"CSV '{file.filename}' uploaded successfully",
        "columns": df.columns.tolist()
    }

# Query request model
class QueryRequest(BaseModel):
    filename: str
    query: str

# Natural Language → Pandas Execution
def run_nl_query(df: pd.DataFrame, query: str):
    schema = ", ".join(df.columns.tolist())

    prompt = f"""
    You are an assistant that converts natural language into pandas code.
    DataFrame is named df. It has columns: {schema}.
    Convert this user request into pandas code that returns a DataFrame:

    User request: "{query}"

    Rules:
    - Always assign the final DataFrame to a variable called result.
    - Use case-insensitive matching for text filters (e.g., str.contains("value", case=False)).
    - If column names are not an exact match, try fuzzy matching with similar words from query.
    - If no filter is detected, just return the first 10 rows with result = df.head(10).
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    code = response.choices[0].message.content.strip("```python").strip("```")

    local_vars = {"df": df.copy(), "pd": pd}
    try:
        exec(code, {}, local_vars)
        result = local_vars.get("result", df.head(10))
    except Exception as e:
        print("⚠️ Error executing generated code:", e)
        result = df.head(10)

    return result

# Query CSV
@app.post("/query_csv")
async def query_csv(request: QueryRequest):
    filename = request.filename
    query = request.query.strip()

    if filename not in CSV_FILES:
        return {"error": f"CSV '{filename}' not found. Please upload it first."}

    df = CSV_FILES[filename]

    # Use GPT to interpret natural language query
    result_df = run_nl_query(df, query)

    return result_df.to_dict(orient="records")
