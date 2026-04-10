from fastapi import APIRouter, HTTPException
from typing import Dict

from src.GenSet import Config, DatasetGenerator
from .schemas import (
    GenerateDatasetRequest,
    GenerateDatasetResponse,
    ConfigResponse,
    HealthResponse
)

router = APIRouter(prefix="/api", tags=["GenSet"])

# Create a single generator instance (shared across requests)
dataset_generator = DatasetGenerator()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Return current configuration and available platforms"""
    return ConfigResponse(
        supported_platforms=list(Config.PLATFORMS.keys()),
        default_models={
            platform: cfg["default_model"] 
            for platform, cfg in Config.PLATFORMS.items()
        },
        loaded_keys={
            "mistral": len(Config.MISTRAL_API_KEYS),
            "ollama": len(Config.OLLAMA_API_KEYS),
            "openai": len(Config.OPENAI_API_KEYS),
            "gemini": len(Config.GEMINI_API_KEYS),
        }
    )


@router.post("/generate", response_model=GenerateDatasetResponse)
async def generate_dataset(request: GenerateDatasetRequest):
    """
    Generate a labeled dataset using GenSet
    """
    try:
        # Validate platform
        if request.platform.lower() not in Config.PLATFORMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported platform '{request.platform}'. "
                       f"Supported: {list(Config.PLATFORMS.keys())}"
            )

        console.print(f"[bold blue]Received request:[/bold blue] {request.num_samples} samples "
                      f"on {request.platform} | Multilingual: {request.multilingual}")

        # Call the generator
        output_file = dataset_generator.create_dataset(
            num_samples=request.num_samples,
            platform=request.platform.lower(),
            language=request.language.lower(),
            labels=request.labels,
            domain=request.domain,
            model=request.model,
            temperature=request.temperature,
            output_file=None,           # Use GenSet default logic
            delay=request.delay,
            multilingual=request.multilingual,
            balance_labels=request.balance_labels,
        )

        return GenerateDatasetResponse(
            status="success",
            message=f"Successfully generated {request.num_samples} samples.",
            output_file=output_file,
            samples_generated=request.num_samples,
            platform_used=request.platform,
            multilingual=request.multilingual,
            file_path=output_file
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate dataset: {str(e)}"
        )