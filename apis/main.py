from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from rich.console import Console

from .routes import router
from src.GenSet import Config

console = Console()

app = FastAPI(
    title="GenSet API",
    description="FastAPI backend for generating classification datasets using Ollama, Mistral, OpenAI, Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "GenSet API is running",
        "documentation": "/docs",
        "health_check": "/api/health",
        "generate_endpoint": "/api/generate"
    }


@app.on_event("startup")
async def startup_event():
    console.print("[bold green]🚀 GenSet FastAPI Server Starting...[/bold green]")
    Config.print_config()


if __name__ == "__main__":
    uvicorn.run("apis.main:app", host="0.0.0.0", port=8000, reload=True)