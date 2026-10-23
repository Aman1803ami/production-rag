import os
import asyncio
import logging
from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional
import litellm
from litellm import Router

from app.config import settings

logger = logging.getLogger(__name__)

# Suppress overly verbose litellm info logs
litellm.suppress_debug_info = True


class LLMRouterService:
    """
    Production LLM Routing and Fallback Service powered by LiteLLM Router.
    Primary: Google Gemini (generous free-tier, low latency)
    Fallback: Free/Local LLM (Groq free tier / Ollama / Local)
    Includes automatic cooldowns, retries, and offline deterministic synthesis fallback.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert AI assistant powering a production enterprise RAG system.\n"
        "Your task is to provide clear, authoritative, and well-structured answers strictly grounded in the provided context.\n\n"
        "Format your answer using clean GitHub-flavored markdown with the following structure:\n"
        "### 🎯 Summary\n"
        "A clear, concise direct answer to the query in 1-3 sentences.\n\n"
        "### 🔍 Key Findings & Analysis\n"
        "- Break down the detailed points into bullet points.\n"
        "- Group related information logically.\n"
        "- Cite sources inline using [Source X] notation where relevant.\n\n"
        "### 💡 Recommendations & Next Steps\n"
        "- Highlight any important caveats, technical specifications, or actionable advice.\n\n"
        "Rules:\n"
        "1. Never invent or assume facts not found in the context.\n"
        "2. If the context is insufficient, state clearly what is known and what is missing."
    )

    def __init__(self):
        self.primary_model = settings.PRIMARY_MODEL
        self.fallback_model = settings.FALLBACK_MODEL
        self.router: Optional[Router] = None
        self._has_valid_keys = False
        self._initialize_router()

    def update_gemini_key(self, api_key: str):
        """Dynamically update Gemini API key and reinitialize router."""
        settings.GEMINI_API_KEY = api_key
        os.environ["GEMINI_API_KEY"] = api_key
        self._initialize_router()
        logger.info("Gemini API key updated. LiteLLM Router reinitialized.")

    def _initialize_router(self):
        """Configure LiteLLM Router with primary and fallback model definitions."""
        model_list: List[Dict[str, Any]] = []

        # 1. Primary Model: Gemini
        gemini_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            model_list.append({
                "model_name": "rag-primary",
                "litellm_params": {
                    "model": self.primary_model,
                    "api_key": gemini_key,
                }
            })
            self._has_valid_keys = True

        # 2. Secondary Model: Groq / Ollama
        groq_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if groq_key and "groq" in self.fallback_model:
            model_list.append({
                "model_name": "rag-fallback",
                "litellm_params": {
                    "model": self.fallback_model,
                    "api_key": groq_key,
                }
            })
            self._has_valid_keys = True
        elif "ollama" in self.fallback_model:
            model_list.append({
                "model_name": "rag-fallback",
                "litellm_params": {
                    "model": self.fallback_model,
                    "api_base": settings.OLLAMA_BASE_URL,
                }
            })
            self._has_valid_keys = True

        if model_list:
            try:
                # Define fallback dictionary
                fallbacks = [{"rag-primary": ["rag-fallback"]}] if len(model_list) > 1 else []
                self.router = Router(
                    model_list=model_list,
                    fallbacks=fallbacks,
                    routing_strategy="simple-shuffle",
                    cooldown_time=60,
                    num_retries=2,
                    timeout=30.0
                )
                logger.info(
                    "LiteLLM Router initialized with models: %s (fallbacks enabled: %s)",
                    [m["model_name"] for m in model_list],
                    bool(fallbacks)
                )
            except Exception as e:
                logger.error("Failed to initialize LiteLLM Router: %s", str(e))
                self.router = None
        else:
            logger.warning(
                "No Gemini or Groq API keys found in environment. "
                "Running in bootstrapping mode: Local contextual synthesizer will handle responses until keys are provided."
            )
            self.router = None

    def _offline_synthesize(self, prompt: str, context: str) -> str:
        """Deterministic context synthesizer used when running in zero-key bootstrapping mode."""
        return (
            f"Based on the indexed document context:\n\n"
            f"{context.strip()}\n\n"
            f"[Note: Answer synthesized via local bootstrapping engine. "
            f"Configure GEMINI_API_KEY in .env to activate full Gemini 2.0 generation with LiteLLM fallback]."
        )

    def _build_messages(
        self,
        query: str,
        context: str,
        custom_system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Construct standard chat completion messages."""
        system_instruction = custom_system_prompt or self.DEFAULT_SYSTEM_PROMPT
        user_content = f"Context Information:\n---------------------\n{context}\n---------------------\n\nUser Question: {query}"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    async def generate_response(
        self,
        query: str,
        context: str,
        custom_system_prompt: Optional[str] = None
    ) -> Tuple[str, str, bool]:
        """
        Generate answer from query and context with automatic fallback.
        Returns: (answer_text, model_name_used, fallback_triggered)
        """
        messages = self._build_messages(query, context, custom_system_prompt)

        if not self.router:
            return self._offline_synthesize(query, context), "local-bootstrapping-synthesizer", False

        try:
            # LiteLLM router handles primary and configured fallback internally
            response = await asyncio.to_thread(
                self.router.completion,
                model="rag-primary",
                messages=messages,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            model_returned = getattr(response, "model", self.primary_model)
            is_fallback = model_returned != self.primary_model and "gemini" not in str(model_returned).lower()
            return answer, model_returned, is_fallback

        except Exception as router_error:
            logger.warning("LiteLLM router execution failed: %s. Using contextual synthesizer.", str(router_error))
            return self._offline_synthesize(query, context), "offline-resilience-fallback", True

    async def stream_response(
        self,
        query: str,
        context: str,
        custom_system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens using LiteLLM with fallback."""
        messages = self._build_messages(query, context, custom_system_prompt)

        if not self.router:
            # Simulate streaming chunks for offline mode
            text = self._offline_synthesize(query, context)
            for w in text.split(" "):
                yield w + " "
            return

        try:
            stream = await asyncio.to_thread(
                self.router.completion,
                model="rag-primary",
                messages=messages,
                max_tokens=1000,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
        except Exception as e:
            logger.warning("Streaming failed (%s). Streaming offline synthesis.", str(e))
            yield self._offline_synthesize(query, context)
