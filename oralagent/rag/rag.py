import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from pydantic import Field

from pydantic import BaseModel, Field
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRerank
# from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
# from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA

# from langchain.memory import ConversationBufferMemory
from langchain_classic.memory import ConversationBufferMemory

from langchain_classic.schema import BaseRetriever
from langchain_classic.docstore.document import Document
from typing import Callable
from datasets import load_dataset
from tqdm import tqdm
import logging
import time

from .qwen3_embedding import Qwen3Embeddings
from .qwen3_rerank import Qwen3Reranker

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Oral 语料向量库子目录名（与 persist_dir 拼接使用）
VECTOR_DB_ORAL_CHINESE_NAME = "vectorDB_OralCorpus_Chinese"
VECTOR_DB_ORAL_ENGLISH_NAME = "vectorDB_OralCorpus_English"


class RAGConfig(BaseModel):
    """Configuration class for RAG (Retrieval Augmented Generation) system.
    Attributes:
        model (str): Cohere model name for chat completion
        temperature (float): Sampling temperature between 0 and 1
        persist_dir (str): Base directory for vector DB; with use_OralCorpus=True, DB path is persist_dir/vectorDB_OralCorpus_*; else DB at persist_dir
        corpus_language (str): "chinese" | "english" — which Oral corpus when use_OralCorpus=True
        embedding_model (str): Cohere model name for embeddings
        rerank_model (str): Cohere model name for reranking
        retriever_k (int): Number of documents to retrieve
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Overlap between text chunks
        local_docs_dir (str): Directory for text files; used for Oral corpus path or custom docs when use_OralCorpus=False
        use_OralCorpus (bool): True = load/use Oral corpus (Chinese/English); False = create from local_docs_dir only
    """

    model: str = Field(default="command-a-03-2025")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    persist_dir: str = Field(default="vector_database", description="Base path for vector DB; subdir added when use_OralCorpus=True")
    corpus_language: Literal["chinese", "english"] = Field(
        default="english",
        description="Which Oral corpus to use: chinese -> vectorDB_OralCorpus_Chinese, english -> vectorDB_OralCorpus_English",
    )
    embedding_model: str = Field(default="embed-english-v3.0")
    rerank_model: str = Field(default="rerank-v3.5")
    retriever_k: int = Field(default=2)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    local_docs_dir: str = Field(default="oralagent/rag/docs")
    use_OralCorpus: bool = Field(default=True)


class RerankingRetriever(BaseRetriever):
    """Custom retriever that wraps a document retrieval function with reranking.
    Attributes:
        get_relevant_docs_func (Callable): Function that retrieves relevant documents
    """

    get_relevant_docs_func: Callable[[str], List[Document]] = Field(...)

    def __init__(self, get_relevant_docs_func: Callable[[str], List[Document]]):
        """Initialize retriever with document retrieval function."""
        super().__init__(get_relevant_docs_func=get_relevant_docs_func)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query.
        Args:
            query (str): Search query
        Returns:
            List[Document]: Retrieved documents
        """
        return self.get_relevant_docs_func(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async retrieval not implemented."""
        raise NotImplementedError("Async retrieval not implemented")


class CohereRAG:
    """RAG system implementation using Cohere's models for embedding, reranking and chat.
    Attributes:
        config (RAGConfig): Configuration for the RAG system
        chat_model (ChatCohere): Cohere chat model
        embeddings (CohereEmbeddings): Cohere embeddings model
        reranker (CohereRerank): Cohere reranking model
        persist_dir (str): Directory for vector database
        memory (ConversationBufferMemory): Conversation memory
        vectorstore (Optional[Chroma]): Vector database for document storage
        local_docs_dir (str): Directory for text files
    """

    def __init__(self, config: RAGConfig = RAGConfig()):
        """Initialize RAG system with given configuration."""
        self.config = config
        self.chat_model = ChatCohere(model=config.model, temperature=config.temperature)
        # self.embeddings = CohereEmbeddings(model=config.embedding_model)
        # self.reranker = CohereRerank(model=config.rerank_model)

        self.embeddings = Qwen3Embeddings(model_name=config.embedding_model)
        self.reranker = Qwen3Reranker(model_name=config.rerank_model)

        # use_OralCorpus=True：在 persist_dir 下使用 vectorDB_OralCorpus_Chinese / vectorDB_OralCorpus_English
        # use_OralCorpus=False：直接使用 persist_dir 作为向量库路径（仅 local_docs_dir 建库）
        if config.use_OralCorpus:
            subdir = (
                VECTOR_DB_ORAL_CHINESE_NAME if config.corpus_language == "chinese" else VECTOR_DB_ORAL_ENGLISH_NAME
            )
            self.persist_dir = os.path.join(config.persist_dir.rstrip(os.sep), subdir)
        else:
            self.persist_dir = config.persist_dir.rstrip(os.sep) if config.persist_dir else config.persist_dir
        self.local_docs_dir = config.local_docs_dir
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="result",
            input_key="query",
        )
        self.vectorstore = self.load_or_create_vectorstore()

        # Initialize vectorstore if empty
        if self.vectorstore is None:
            all_documents = []

            if self.config.use_OralCorpus:
                # 使用 Oral 语料：按语种加载 vectorDB_OralCorpus_Chinese 或 vectorDB_OralCorpus_English 对应数据
                if self.config.corpus_language == "chinese":
                    print("Loading documents from OralCorpus (Chinese)...")
                    oral_docs = self.load_OralCorpus_Chinese(self.local_docs_dir)
                    all_documents.extend(oral_docs)
                    print(f"Loaded {len(oral_docs)} documents from OralCorpus (Chinese)")
                else:
                    print("Loading documents from OralCorpus (English)...")
                    oral_docs = self.load_OralCorpus_English(self.local_docs_dir)
                    all_documents.extend(oral_docs)
                    print(f"Loaded {len(oral_docs)} documents from OralCorpus (English)")
                # 可选：同时加载 local_docs_dir 下的文档
                if os.path.exists(self.local_docs_dir):
                    print(f"Loading documents from local directory: {self.local_docs_dir}")
                    local_docs = self.load_directory(self.local_docs_dir)
                    all_documents.extend(local_docs)
                    print(f"Loaded {len(local_docs)} documents from local directory")
            else:
                # 仅 local_docs_dir 逻辑：从本地目录建库，向量库路径为 persist_dir
                if os.path.exists(self.local_docs_dir):
                    print(f"Loading documents from local directory: {self.local_docs_dir}")
                    local_docs = self.load_directory(self.local_docs_dir)
                    all_documents.extend(local_docs)
                    print(f"Loaded {len(local_docs)} documents from local directory")

            if all_documents:
                print(f"Creating vectorstore with {len(all_documents)} total documents")
                self.create_or_update_vectorstore(all_documents)
            else:
                print("Warning: No documents loaded. Please check your configuration.")

    def load_directory(self, directory_path: str) -> List[Document]:
        """Load and split all .txt files from a directory into documents.
        Args:
            directory_path (str): Path to directory containing text files
        Returns:
            List[Document]: List of processed documents
        Raises:
            ValueError: If directory does not exist
        """
        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")

        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )

        # Process each document file (txt, pdf, docx)
        for file_pattern in ["**/*.txt", "**/*.pdf", "**/*.docx"]:
            for file_path in directory.glob(file_pattern):
                try:
                    # Select appropriate loader based on file extension
                    if file_path.suffix.lower() == ".txt":
                        loader = TextLoader(str(file_path))
                    elif file_path.suffix.lower() == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == ".docx":
                        loader = Docx2txtLoader(str(file_path))

                    docs = loader.load()

                    # Split documents first
                    split_docs = text_splitter.split_documents(docs)

                    # Add metadata to each document
                    metadata = {
                        "source": str(file_path),
                        "created_at": os.path.getctime(file_path),
                        "file_type": file_path.suffix.lower()[1:],
                    }

                    for doc in split_docs:
                        doc.metadata.update(metadata)

                    documents.extend(split_docs)
                    print(f"Loaded and split: {file_path}")

                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

        return documents

    def load_or_create_vectorstore(self) -> Optional[Chroma]:
        """Load existing vectorstore or prepare for new one.
        Returns:
            Optional[Chroma]: Loaded vectorstore or None if not exists
        """
        if os.path.exists(self.persist_dir):
            print("Loading existing vectorstore...")
            return Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        return None

    def create_or_update_vectorstore(self, documents: List[Document]):
        """Create new vectorstore or add documents to existing one.
        Args:
            documents (List[Document]): Documents to add to vectorstore
        """
        if self.vectorstore is None:
            print("Creating new vectorstore...")
            # documents = documents[:500]  # Limit to first 1000 documents for initial creation # note here !!!!

            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
            # 分批添加文档
            batch_size = 1  # 每批处理 10 个文档
            batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

            for batch in tqdm(batches, desc="Adding documents to vectorstore", unit="batch"):
                self.vectorstore.add_documents(batch)
                # time.sleep(1)  # 每批之间等待 1 秒，避免触发速率限制

            # self.vectorstore = Chroma.from_documents(
            #     documents=documents,
            #     embedding=self.embeddings,
            #     persist_directory=self.persist_dir,
            # )
        else:
            print("Adding documents to existing vectorstore...")
            self.vectorstore.add_documents(documents)

        print(f"Vectorstore saved to {self.persist_dir}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using vector similarity and reranking.
        Args:
            query (str): Search query
        Returns:
            List[Document]: Reranked relevant documents
        """
        logging.info(f" ### RAG query: {query}")
        logging.info(f" ### Number of documents in vectorstore: {self.vectorstore._collection.count()}")
        logging.info(f" ### Retriever k value: {self.config.retriever_k * 2}")
        
        # Get initial candidates using vector similarity
        docs = self.vectorstore.similarity_search(query, k=self.config.retriever_k * 2)
        logging.info(f" ### Retrieved docs: {docs}")

        if not docs:
            logging.error("No documents retrieved from similarity search.")
        else:
            logging.info(f"Number of documents retrieved: {len(docs)}")
        
        # # 打印检索出的文档内容
        # logging.info("Retrieved documents (before reranking):")
        # for i, doc in enumerate(docs):
        #     logging.info(f"Doc {i + 1}: {doc.page_content}")

        # Rerank documents
        reranked = self.reranker.rerank(query=query, documents=[doc.page_content for doc in docs])

        # # 打印重排序后的文档内容
        # logging.info("Reranked documents:")
        # for i, doc in enumerate(reranked):
        #     logging.info(f"Doc {i + 1}: {doc}")

        # Return top k documents after reranking
        return [docs[result["index"]] for result in reranked[: self.config.retriever_k]]

    def initialize_rag(self, with_memory: bool = False) -> RetrievalQA:
        """Initialize RAG chain with optional conversation memory.
        Args:
            with_memory (bool): Whether to include conversation memory
        Returns:
            RetrievalQA: Configured RAG chain
        Raises:
            ValueError: If vectorstore not initialized
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please add documents first.")

        # Create custom retriever
        retriever = RerankingRetriever(self.get_relevant_documents)

        # Configure chain parameters
        chain_kwargs = {
            "llm": self.chat_model,
            "chain_type": "stuff",
            "retriever": retriever,
            "return_source_documents": True,
            "verbose": True,
        }

        if with_memory:
            chain_kwargs["memory"] = self.memory

        return RetrievalQA.from_chain_type(**chain_kwargs)

    ## Add by Bryce
    def initialize_rag_wo_llm(self, query: str) -> List[dict]:
        """Initialize RAG chain with optional conversation memory.
        Args:
            with_memory (bool): Whether to include conversation memory
        Returns:
            RetrievalQA: Configured RAG chain
        Raises:
            ValueError: If vectorstore not initialized
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please add documents first.")

        # Create custom retriever
        retriever_content = self.get_relevant_documents(query)
        
        # 重新整理数据格式
        formatted_result = [
            {
                "content": doc.page_content,  # 文档内容
                "source": doc.metadata.get("source", "unknown")  # 从 metadata 中获取 source 字段，若不存在则返回 "unknown"
            }
            for doc in retriever_content
        ]

        return formatted_result

    def load_medrag_textbooks(self) -> List[Document]:
        """Load MedRAG textbooks dataset from Hugging Face.
        Returns:
            List[Document]: List of processed documents from MedRAG textbooks
        Raises:
            ValueError: If unable to load the dataset
        """
        try:
            print("Loading MedRAG textbooks dataset...")
            dataset = load_dataset("MedRAG/textbooks", split="train")
            documents = []

            for item in tqdm(
                dataset, desc="Processing MedRAG textbooks", total=len(dataset), unit="chunk"
            ):
                # Create a Document object for each textbook snippet
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "source": f"MedRAG/textbooks",
                        "id": item["id"],
                        "title": item["title"],
                    },
                )
                documents.append(doc)

            print(f"Loaded {len(documents)} document chunks from MedRAG textbooks")
            return documents

        except Exception as e:
            print(f"Error loading MedRAG textbooks: {str(e)}")
            raise ValueError(f"Failed to load MedRAG textbooks dataset: {str(e)}")

    def load_OralCorpus_Chinese(self, corpus_path: str) -> List[Document]:
        """Load OralCorpus dataset from local directory containing jsonl files (Chinese fields).
        Each jsonl line should have: ID, 学科, 学科_ID, 书名, 页码, 内容, 原版语言.
        Args:
            corpus_path (str): Local path to directory containing .jsonl files
        Returns:
            List[Document]: List of Document objects for RAG
        Raises:
            ValueError: If path does not exist or is not a directory
        """
        corpus_dir = Path(corpus_path)
        if not corpus_dir.exists():
            raise ValueError(f"Corpus path does not exist: {corpus_path}")
        if not corpus_dir.is_dir():
            raise ValueError(f"Corpus path is not a directory: {corpus_path}")

        documents = []
        jsonl_files = list(corpus_dir.glob("**/*.jsonl"))

        for file_path in tqdm(jsonl_files, desc="Loading OralCorpus jsonl files (Chinese)", unit="file"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            content = item.get("内容", "")
                            if not content:
                                continue
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": item.get("书名", "") + ", pp. " + str(item.get("页码", "")) + ".",
                                    "id": item.get("ID"),
                                    "学科": item.get("学科", ""),
                                    "学科_ID": item.get("学科_ID", ""),
                                    "书名": item.get("书名", ""),
                                    "页码": item.get("页码"),
                                    "原版语言": item.get("原版语言", ""),
                                },
                            )
                            documents.append(doc)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skip invalid JSON at {file_path}:{line_num}: {e}")
                            continue
            except Exception as e:
                logging.warning(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(documents)} document chunks from OralCorpus Chinese ({len(jsonl_files)} files)")
        return documents

    def load_OralCorpus_English(self, corpus_path: str) -> List[Document]:
        """Load OralCorpus dataset from local directory containing jsonl files.
        Each jsonl line should have: ID, 学科, 学科_ID, 书名, 页码, 内容, 原版语言.
        Args:
            corpus_path (str): Local path to directory containing .jsonl files
        Returns:
            List[Document]: List of Document objects for RAG
        Raises:
            ValueError: If path does not exist or is not a directory
        """
        corpus_dir = Path(corpus_path)
        if not corpus_dir.exists():
            raise ValueError(f"Corpus path does not exist: {corpus_path}")
        if not corpus_dir.is_dir():
            raise ValueError(f"Corpus path is not a directory: {corpus_path}")

        documents = []
        jsonl_files = list(corpus_dir.glob("**/*.jsonl"))

        for file_path in tqdm(jsonl_files, desc="Loading OralCorpus jsonl files", unit="file"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            content = item.get("Content", "")
                            if not content:
                                continue
                            doc = Document(
                                page_content=content,
                                metadata={ 
                                    "source": item.get("Title", "") + ", pp. " + str(item.get("Page_number", "")) + ".",
                                    "id": item.get("ID"),
                                    "Subject": item.get("Subject", ""),
                                    "Subject_ID": item.get("Subject_ID", ""),
                                    "Title": item.get("Title", ""),
                                    "Page_number": item.get("Page_number"),
                                    "Init_lang": item.get("Init_lang", ""),
                                },
                            )
                            documents.append(doc)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skip invalid JSON at {file_path}:{line_num}: {e}")
                            continue
            except Exception as e:
                logging.warning(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(documents)} document chunks from OralCorpus ({len(jsonl_files)} files)")
        return documents