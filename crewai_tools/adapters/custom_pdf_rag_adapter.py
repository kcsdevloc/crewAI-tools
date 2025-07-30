from typing import Any, Optional

from crewai_tools.rag.core import CustomRAGAdapter as BaseCustomRAGAdapter
from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import Adapter


class CustomPDFRAGAdapter(Adapter):
    """Custom PDF RAG adapter that replaces PDFEmbedchainAdapter."""

    summarize: bool = False
    src: Optional[str] = None

    def __init__(
        self,
        collection_name: str = "crewai_pdf_knowledge_base",
        persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        summarize: bool = False,
        top_k: int = 5,
        embedding_api_key: Optional[str] = None,
        **embedding_kwargs
    ):
        self.summarize = summarize

        # Prepare embedding configuration
        embedding_config = {
            "model": embedding_model,
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
        where_filter = None
        if self.src:
            where_filter = {"source": self.src}

        result = self._adapter.query(question, where=where_filter)

        if self.summarize:
            return result
        else:
            # Extract just the document content for compatibility
            lines = result.split('\n\n')
            sources = []
            for line in lines:
                if line.startswith('[Source:'):
                    # Extract content after the source line
                    content_start = line.find('\n')
                    if content_start != -1:
                        sources.append(line[content_start + 1:])
                else:
                    sources.append(line)
            return "\n\n".join(sources)

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add PDF content to the knowledge base."""
        self.src = args[0] if args else None

        # Use PDF processor by setting data_type
        kwargs['data_type'] = DataType.PDF

        self._adapter.add(*args, **kwargs)
