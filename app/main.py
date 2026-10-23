import os
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.dependencies import initialize_services, shutdown_services
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("rag_application")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook: manages startup and graceful shutdown."""
    logger.info("Starting Production RAG Application in %s mode...", settings.ENVIRONMENT)
    await initialize_services()
    yield
    logger.info("Shutting down Production RAG Application...")
    await shutdown_services()


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Production RAG API",
        version="1.0.0",
        description="High-performance Production RAG system featuring Qdrant, Two-Tier Redis Caching, FlashRank Reranker, and Resilient LiteLLM Routing.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and Latency Middleware
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:10]}"
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error("[%s] Unhandled exception: %s", req_id, str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "request_id": req_id,
                "error": "InternalServerError",
                "message": "An unexpected error occurred during processing.",
                "details": str(exc)
            }
        )

    # Register API Routes
    app.include_router(api_router)

    # Static Files and UI Playground
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_ui():
            index_file = os.path.join(static_dir, "index.html")
            return FileResponse(index_file)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=(settings.ENVIRONMENT == "development")
    )
