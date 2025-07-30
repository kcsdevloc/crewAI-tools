from typing import Any, Optional

from crewai_tools.rag.core import CustomRAGAdapter as BaseCustomRAGAdapter
from crewai_tools.tools.rag.rag_tool import Adapter


class CustomRAGAdapter(Adapter):
    """Custom RAG adapter that replaces EmbedchainAdapter."""

    summarize: bool = False

    def __init__(
        self,
        collection_name: str = "crewai_knowledge_base",
        persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        summarize: bool = False,
        top_k: int = 5,
        embedding_api_key: Optional[str] = None,
        **embedding_kwargs
    ):
        super().__init__()
        self.summarize = summarize

        # Prepare embedding configuration
        embedding_config = {
            "api_key": embedding_api_key,
            **embedding_kwargs
        }

        self._adapter = BaseCustomRAGAdapter(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            summarize=summarize,
            top_k=top_k,
            embedding_config=embedding_config
        )

    def query(self, question: str) -> str:
        """Query the knowledge base with a question and return the answer."""
        return self._adapter.query(question)

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add content to the knowledge base."""
        self._adapter.add(*args, **kwargs)
