import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

import train as train_module
import model_manager
from dqn_agent import DQNAgent

# Import your SearchAI backend
import main


app = FastAPI()

# Initialize the core filesystem engine
filesystem = main.SearchAIFileSystem(main.MODEL_DIR)


# ============================================================
# DATA SCHEMAS
# ============================================================

class CreateModelPayload(BaseModel):
    name: str
    engine: str
    query: str
    mode: str
    criteria: list[str]
    input_size: int
    hidden_layers: list[int]
    output_size: int


class TrainInitPayload(BaseModel):
    model_name: str


class TrainCompletePayload(BaseModel):
    model_name: str
    training_data: list


# ============================================================
# MODEL CREATION
# ============================================================

@app.post("/api/create")
async def create_model(payload: CreateModelPayload):
    try:
        result = filesystem.create_model_headless(
            name=payload.name,
            engine=payload.engine,
            query=payload.query,
            mode=payload.mode,
            criteria=payload.criteria,
            input_size=payload.input_size,
            hidden_layers=payload.hidden_layers,
            output_size=payload.output_size,
        )

        return result

    except ValueError as e:
        return {
            "status": "error",
            "message": str(e)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create model: {e}"
        }


# ============================================================
# FOLDER CREATION
# ============================================================

@app.post("/api/mkdir")
async def create_folder(data: dict):
    try:
        name = data.get("name", "").strip()

        result = filesystem.create_folder_headless(name)

        return result

    except ValueError as error:
        return {
            "status": "error",
            "message": str(error)
        }

    except Exception as error:
        return {
            "status": "error",
            "message": f"Failed to create folder: {error}"
        }


# ============================================================
# TRAINING
# ============================================================

@app.post("/api/train/init")
async def init_train(payload: TrainInitPayload):
    try:
        data = train_module.init_training_session(payload.model_name)

        return {
            "status": "success",
            "data": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/api/train/complete")
async def complete_train(payload: TrainCompletePayload):
    try:
        res = train_module.complete_training_session(
            payload.model_name,
            payload.training_data
        )

        return res

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# STATIC FILE SERVING
# ============================================================

@app.get("/")
async def serve_html():
    return FileResponse("index.html")


@app.get("/style.css")
async def serve_css():
    return FileResponse("style.css")


@app.get("/script.js")
async def serve_js():
    return FileResponse("script.js")


# ============================================================
# MODEL EXPLORER
# ============================================================

@app.get("/models")
async def list_models(path: str = ""):
    try:
        base_models_dir = model_manager.get_models_dir()
        target_dir = (base_models_dir / path).resolve()

        if not str(target_dir).startswith(
            str(base_models_dir.resolve())
        ):
            return []

        if not target_dir.exists() or not target_dir.is_dir():
            return []

        items = []

        for entry in target_dir.iterdir():
            if entry.is_dir():
                is_model = (
                    (entry / "metadata.json").exists()
                    or (entry / "criteria.json").exists()
                )

                items.append({
                    "name": entry.name,
                    "type": "model" if is_model else "folder",
                    "relative_path": str(
                        entry.relative_to(base_models_dir)
                    ).replace("\\", "/")
                })

            elif entry.is_file() and entry.name == "model.pt":
                items.append({
                    "name": entry.name,
                    "type": "model",
                    "relative_path": str(
                        entry.relative_to(base_models_dir)
                    ).replace("\\", "/")
                })

        return items

    except Exception:
        return []


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )