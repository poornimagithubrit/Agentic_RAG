import os
import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from pydantic import BaseModel
import re

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

# Query CSV
class QueryRequest(BaseModel):
    filename: str
    query: str
@app.post("/query_csv")
async def query_csv(request: QueryRequest):
    filename = request.filename
    query = request.query.lower().strip()

    if filename not in CSV_FILES:
        return {"error": f"CSV '{filename}' not found. Please upload it first."}

    df = CSV_FILES[filename]

    # Step 1: Detect filters dynamically (looser regex)
    filters = []
    for col in df.columns:
        col_lower = col.lower()

        # Allow "serial number A389754", "serial number is A389754", etc.
        pattern = rf"{col_lower}\s*(?:is|=|for)?\s*([\w\s\d\.-]+)"
        match = re.search(pattern, query)
        if match:
            filters.append((col, match.group(1).strip()))

    # Apply filters
    filtered_df = df.copy()
    for col, value in filters:
        filtered_df = filtered_df[filtered_df[col].astype(str).str.lower() == value.lower()]

    # Step 2: Detect explicitly requested columns
    requested_cols = []
    for col in df.columns:
        if col.lower() in query and all(fcol.lower() != col.lower() for fcol, _ in filters):
            requested_cols.append(col)

    # Step 3: Columns to return
    cols_to_return = [f[0] for f in filters] + requested_cols
    cols_to_return = list(dict.fromkeys(cols_to_return))  # remove duplicates

    # Step 4: Response
    if not filtered_df.empty:
        if cols_to_return:
            result = filtered_df[cols_to_return].to_dict(orient="records")
        else:
            result = filtered_df[[f[0] for f in filters]].to_dict(orient="records") if filters else df.head(5).to_dict(orient="records")
    else:
        # Keyword fallback
        matched = False
        for word in query.split():
            mask = pd.DataFrame(False, index=df.index, columns=df.columns)
            for col in df.columns:
                mask[col] = df[col].astype(str).str.lower().str.contains(word)
            combined_mask = mask.any(axis=1)
            if combined_mask.any():
                filtered_df = df[combined_mask]
                result = filtered_df.head(10).to_dict(orient="records")
                matched = True
                break

        if not matched:
            result = df.head(10).to_dict(orient="records")

    return result
