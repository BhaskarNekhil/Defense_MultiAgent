"""
Defense-AI FastAPI Server
Implements the full OpenEnv interface: /reset, /step, /state, /health
"""
import sys
import os
import traceback

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Any, Dict

from models import DefenseAction, DefenseObservation
from defense_env.environment import DefenseEnvironment

_env: Optional[DefenseEnvironment] = None

app = FastAPI(
    title="Defense-AI",
    description="Air Defense OpenEnv Environment — OpenEnv spec compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("\n=== SERVER ERROR ===")
    print(tb)
    print("===================\n")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": tb},
        headers={"Access-Control-Allow-Origin": "*"},
    )





class StepRequest(BaseModel):
    action:            Optional[Any]                  = None
    mission_id:        Optional[str]                  = None
    session_id:        Optional[str]                  = None
    action_type:       Optional[str]                  = None
    target_id:         Optional[str]                  = None
    decisions_summary: Optional[str]                  = None
    engagements:       Optional[List[Dict[str, str]]] = None
    model_config = {"extra": "allow"}

    def get_action(self) -> DefenseAction:
        if self.action and isinstance(self.action, dict):
            return DefenseAction(**self.action)
        return DefenseAction(
            action_type       = self.action_type or "scan_airspace",
            target_id         = self.target_id,
            decisions_summary = self.decisions_summary,
            engagements       = self.engagements,
        )


@app.post("/reset")
async def reset(request: Request):
    global _env
    task_id = "task_easy"
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = body.get("task_id", "task_easy") or "task_easy"
    except Exception:
        pass
    _env = DefenseEnvironment(task_id=task_id)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
async def step(req: StepRequest):
    global _env
    if _env is None:
        _env = DefenseEnvironment()
        _env.reset()
    obs = _env.step(req.get_action())
    return obs.model_dump()


@app.get("/state")
async def state():
    global _env
    if _env is None:
        return {"error": "No active mission. Call /reset first."}
    return _env.state.model_dump()


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "Defense-AI", "version": "1.0.0"}


_server_dir = os.path.dirname(os.path.abspath(__file__))
_static_dir = os.path.join(_server_dir, "static")

if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    @app.get("/")
    async def serve_ui():
        return FileResponse(os.path.join(_static_dir, "index.html"))
else:
    @app.get("/")
    async def serve_root():
        return {"name": "Defense-AI", "docs": "/docs"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

if __name__ == "__main__":
    main()