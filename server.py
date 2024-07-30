import argparse
import uvicorn
import json

from copy import deepcopy
from typing import List, Dict, Any

from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from sum_model import sum

import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

app = FastAPI()

class Item(BaseModel):
    src: str
    sum_rate: float



@app.get("/")
async def root():
    return {"message": "Welcome to Summarization Server!"}


@app.post("/summarize")
async def summarize(request: Item):
    # image = image.file
    # image = Image.open(image).convert('RGB')
    # res = OCR_MODEL.ocr(image)
    src = request.src
    sum_rate = request.sum_rate
    
    sum_result = sum(src, sum_rate)
    ret = {"sum": sum_result}
    return ret



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='127.0.0.1')
    parser.add_argument("--port", type=int, default=8000)

    tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/user/.cache/huggingface/hub/models--gogamza--kobart-summarization/snapshots/31f181b155a0ad74bd93bd90ee04310ff72691f4')
    model = BartForConditionalGeneration.from_pretrained('/home/user/.cache/huggingface/hub/models--gogamza--kobart-summarization/snapshots/31f181b155a0ad74bd93bd90ee04310ff72691f4')
    args = parser.parse_args()
    uvicorn.run(app,
                host=args.host,
                port=args.port)