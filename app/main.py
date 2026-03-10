# app/main.py
from fastapi import FastAPI, Form, File, UploadFile
from PIL import Image
import uvicorn
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from contextlib import asynccontextmanager

app = FastAPI(title="Florence-2 Segmentation Service")


# Глобальные переменные
model = None
processor = None
device = "cuda"
caption_tasks = {"<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor

    print(f"Loading Florence-2 on {device} ...")

    model_id = "microsoft/Florence-2-large"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,  # ← только base, НЕ large!
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda",  # критично для маленькой VRAM
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print("Model loaded")
    yield

    # cleanup
    model = None
    processor = None
    torch.cuda.empty_cache()


app.router.lifespan_context = lifespan


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    task: str = Form("<REFERRING_EXPRESSION_SEGMENTATION>"),
    text_input: str = Form("all buildings"),
    only_task: bool = Form(False),
):
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Invalid image"}

        original_size = (img.shape[1], img.shape[0])  # ширина, высота оригинала

        pil_img = Image.fromarray(img)
        pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)  # уменьши для 4 ГБ VRAM

        if only_task:
            prompt = task
        else:
            prompt = task + text_input

        inputs = processor(text=prompt, images=pil_img, return_tensors="pt")
        inputs["pixel_values"] = inputs[
            "pixel_values"
        ].half()  # исправление dtype mismatch
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=original_size,  # ← передаём размер оригинального изображения!
        )

        text = processor.batch_decode(
            generated_ids, skip_special_tokens=True  # ← ВАЖНО
        )[0]

        return {
            "task": task,
            "result": parsed,  # строка или dict — как пришло
            "raw_generated_text": generated_text,
            "clean_text": text.strip(),
            "type": "caption" if task in caption_tasks else "structured"
        }

    except Exception as e:
        return {"error": str(e)}
