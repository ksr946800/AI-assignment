# server.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import sqlite3

app = FastAPI()

# Load model
tokenizer = BartTokenizer.from_pretrained("sql_bart_model")
model = BartForConditionalGeneration.from_pretrained("sql_bart_model")
model.eval()

# Templates and static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "sql": "", "result": ""})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, query: str = Form(...)):
    input_text = "translate English to SQL: " + query
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=64, truncation=True)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=64)

    sql = tokenizer.decode(output[0], skip_special_tokens=True)

    # Optional: run against SQLite DB
    try:
        conn = sqlite3.connect("sample.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        result = rows
    except Exception as e:
        result = f"Error executing SQL: {e}"
    finally:
        conn.close()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "sql": sql,
        "result": result,
        "query": query
    })
