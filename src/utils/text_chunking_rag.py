# --- START OF FILE text_chunking_rag.py ---

import os
import json
from typing import List, Dict, Optional, Any, Tuple
import logging
from pathlib import Path

# Import shared utilities
from src.utils.utils import get_device, get_torch_dtype

# Langchain and related imports with error handling
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(
        f"Langchain components not found: {e}. Please install required packages:")
    print("pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers")
    LANGCHAIN_AVAILABLE = False

# LLM imports with error handling
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers or PyTorch not found: {e}. LLM features disabled.")
    # bitsandbytes for 8/4bit
    print("pip install transformers torch accelerate bitsandbytes")
    TRANSFORMERS_AVAILABLE = False
    HuggingFacePipeline = None  # Define as None if import fails

logger = logging.getLogger(__name__)


class TextChunkingRAG:
    """
    Handles text chunking, embedding creation using Sentence Transformers,
    FAISS vector store management, and Retrieval-Augmented Generation (RAG)
    for question answering, with support for CUDA and MPS devices.
    """

    def __init__(
        self,
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store_dir: Optional[str | Path] = None,
        llm_model_name: Optional[str] = None,
        llm_max_new_tokens: int = 512,
        llm_temperature: float = 0.1,
        use_bf16_if_available: bool = True  # Prefer bfloat16 on compatible GPUs
    ):
        """
        Initialize the TextChunkingRAG system.

        Args:
            embeddings_model_name (str): HF model name for embeddings.
            chunk_size (int): Target size for text chunks.
            chunk_overlap (int): Overlap between consecutive chunks.
            vector_store_dir (str | Path, optional): Directory to save/load FAISS vector stores.
            llm_model_name (str, optional): HF model name for the generation LLM. If None,
                                            only retrieval is performed.
            llm_max_new_tokens (int): Max tokens for the LLM to generate.
            llm_temperature (float): Sampling temperature for LLM generation.
            use_bf16_if_available (bool): Use bfloat16 for LLM if supported, else float16/float32.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "Langchain components are required. Please install dependencies.")

        self.device = get_device()
        self.embeddings_model_name = embeddings_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_dir = Path(
            vector_store_dir) if vector_store_dir else None
        self.llm_model_name = llm_model_name
        self.llm_max_new_tokens = llm_max_new_tokens
        self.llm_temperature = llm_temperature
        self.use_bf16_if_available = use_bf16_if_available

        # Initialize embeddings on the detected device
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model_name,
            # Pass 'cuda', 'mps', or 'cpu' string
            model_kwargs={'device': self.device.type}
        )
        logger.info(
            f"Initialized embeddings model '{embeddings_model_name}' on device '{self.device.type}'")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[HuggingFacePipeline] = None
        self.rag_chain = None

        if llm_model_name:
            if not TRANSFORMERS_AVAILABLE or HuggingFacePipeline is None:
                logger.warning(
                    "Transformers/PyTorch not available. Cannot initialize LLM.")
                self.llm_model_name = None  # Disable LLM features
            else:
                self._initialize_llm(llm_model_name)

        if self.vector_store_dir:
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Vector store directory set to: {self.vector_store_dir}")

    def _initialize_llm(self, model_name: str):
        """Initialize the HuggingFace pipeline LLM."""
        logger.info(
            f"Initializing LLM: {model_name} on device {self.device}...")
        try:
            # Determine appropriate dtype
            if self.device.type == 'cuda':
                if self.use_bf16_if_available and torch.cuda.get_device_capability(self.device)[0] >= 8:
                    torch_dtype = torch.bfloat16
                    logger.info("Using torch.bfloat16 for LLM.")
                else:
                    torch_dtype = torch.float16
                    logger.info("Using torch.float16 for LLM.")
            elif self.device.type == 'mps':
                # Float16 can be problematic on MPS, often safer to use float32 unless tested
                torch_dtype = torch.float16  # Or torch.float32 if issues arise
                logger.info(
                    "Using torch.float16 for LLM on MPS. (Consider torch.float32 if encountering issues)")
            else:
                torch_dtype = torch.float32
                logger.info("Using torch.float32 for LLM on CPU.")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                # device_map='mps' can be buggy
                device_map=self.device.type if self.device.type != 'mps' else 'auto',
                # For MPS, 'auto' might be safer or manually place layers if needed
                # load_in_8bit=True, # Optional: if you need 8-bit quantization
                # load_in_4bit=True, # Optional: if you need 4-bit quantization
            )
            # Ensure model is on the correct device if device_map wasn't used or failed (esp. for MPS)
            if self.device.type == 'mps':
                model.to(self.device)

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=self.device,  # Explicitly set device for pipeline
                max_new_tokens=self.llm_max_new_tokens,
                temperature=self.llm_temperature,
                do_sample=True if self.llm_temperature > 0 else False,
                # top_p=0.9, # Optional: nucleus sampling
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"LLM '{model_name}' initialized successfully.")

        except ImportError:
            logger.error(
                "Failed to import transformers components. LLM initialization skipped.")
            self.llm = None
        except Exception as e:
            logger.error(
                f"Error initializing LLM '{model_name}': {e}", exc_info=True)
            logger.error("Troubleshooting tips:")
            logger.error("- Ensure sufficient GPU memory.")
            logger.error("- Check model name and HF Hub accessibility.")
            logger.error(
                "- Try updating transformers: pip install --upgrade transformers")
            logger.error(
                "- If using quantization (4/8bit), ensure bitsandbytes is installed.")
            logger.error(
                "- For MPS issues, try torch_dtype=torch.float32 or update PyTorch nightly.")
            self.llm = None  # Fallback to retrieval-only

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """Split text into chunks."""
        documents = self.text_splitter.create_documents(
            [text], metadatas=[metadata] if metadata else None)
        logger.info(
            f"Split text into {len(documents)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap}).")
        return documents

    def create_vector_store(self, documents: List[Document], document_id: str) -> Optional[FAISS]:
        """Create and optionally save a FAISS vector store."""
        if not documents:
            logger.warning("No documents provided to create vector store.")
            return None
        try:
            self.vector_store = FAISS.from_documents(
                documents, self.embeddings)
            logger.info(
                f"Created FAISS vector store with {len(documents)} embeddings.")

            if self.vector_store_dir:
                save_path = self.vector_store_dir / document_id
                # FAISS save_local expects directory path, not index file path
                save_path.mkdir(parents=True, exist_ok=True)
                # Pass directory path as string
                self.vector_store.save_local(str(save_path))
                logger.info(f"Vector store saved locally to: {save_path}")

            return self.vector_store
        except Exception as e:
            logger.error(
                f"Error creating or saving vector store for '{document_id}': {e}", exc_info=True)
            self.vector_store = None
            return None

    def load_vector_store(self, document_id: str) -> bool:
        """Load a FAISS vector store from the specified directory."""
        if not self.vector_store_dir:
            logger.error(
                "Cannot load vector store: vector_store_dir not configured.")
            return False

        load_path = self.vector_store_dir / document_id
        # FAISS load_local expects directory path
        if not load_path.is_dir():
            logger.warning(f"Vector store directory not found at: {load_path}")
            return False

        try:
            # FAISS requires allow_dangerous_deserialization with newer langchain/pickle
            self.vector_store = FAISS.load_local(
                str(load_path),  # Pass directory path as string
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Successfully loaded vector store from: {load_path}")
            return True
        except ModuleNotFoundError as e:
            logger.error(
                f"Error loading vector store from {load_path}: Missing module '{e.name}'. Ensure all necessary classes are defined/imported.")
            self.vector_store = None
            return False
        except Exception as e:
            logger.error(
                f"Error loading vector store from {load_path}: {e}", exc_info=True)
            self.vector_store = None
            return False

    def process_document(self, text: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Chunk text, create embeddings, and build/save the vector store.
        Checks if store exists first.

        Args:
            text (str): The document text content.
            document_id (str): A unique identifier for the document (e.g., filename stem).

        Returns:
            Dict[str, Any]: Info about the processing, or None if failed.
        """
        if self.vector_store_dir and self.load_vector_store(document_id):
            logger.info(
                f"Using existing vector store for document ID: {document_id}")
            # Optionally, return info about the loaded store
            return {
                "document_id": document_id,
                "status": "loaded_existing_store",
                "vector_store_path": str(self.vector_store_dir / document_id)
            }

        logger.info(f"Processing document ID: {document_id}")
        # Chunk the text
        metadata = {"source": document_id}  # Add source metadata to chunks
        documents = self.chunk_text(text, metadata=metadata)
        if not documents:
            logger.error(
                f"Text chunking resulted in zero documents for {document_id}")
            return None

        # Create and save vector store
        vector_store = self.create_vector_store(documents, document_id)
        if not vector_store:
            logger.error(
                f"Failed to create vector store for document ID: {document_id}")
            return None

        # Return processing results
        result = {
            "document_id": document_id,
            "status": "created_new_store",
            "text_length": len(text),
            "chunk_count": len(documents),
            "vector_store_type": "FAISS",
        }
        if self.vector_store_dir:
            result["vector_store_path"] = str(
                self.vector_store_dir / document_id)

        return result

    def setup_rag_chain(self, k: int = 4):
        """Set up the Langchain RAG chain."""
        if not self.vector_store:
            logger.error(
                "Cannot set up RAG chain: Vector store not initialized.")
            return
        if not self.llm:
            logger.warning(
                "LLM not initialized. RAG chain setup skipped. Will perform retrieval only.")
            return

        if self.rag_chain:
            logger.info("RAG chain already set up.")
            return

        logger.info("Setting up RAG chain...")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        template = """
        You are an AI assistant specialized in analyzing legal contracts.
        Use the following pieces of retrieved context from a contract to answer the question accurately and concisely.
        If the answer cannot be found in the provided context, state that clearly.
        Do not make up information.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info(f"RAG chain set up successfully with retriever k={k}.")

    def answer_question(self, question: str, k: int = 4) -> str:
        """
        Answer a question using retrieval or the full RAG chain if available.

        Args:
            question (str): The question to answer.
            k (int): Number of relevant chunks to retrieve.

        Returns:
            str: The answer or retrieved context.
        """
        if not self.vector_store:
            return "Error: Vector store not initialized. Please process a document first."

        if self.llm:
            # Use the RAG chain
            if not self.rag_chain:
                self.setup_rag_chain(k=k)  # Setup if not already done

            if self.rag_chain:  # Check if setup was successful
                try:
                    logger.info(
                        f"Answering question using RAG chain (k={k}): '{question}'")
                    answer = self.rag_chain.invoke(question)
                    return answer
                except Exception as e:
                    logger.error(
                        f"Error invoking RAG chain: {e}", exc_info=True)
                    logger.warning(
                        "Falling back to simple retrieval due to RAG chain error.")
            else:
                logger.warning(
                    "RAG chain setup failed. Falling back to simple retrieval.")

        # Fallback or retrieval-only mode
        logger.info(
            f"Performing similarity search (k={k}) for question: '{question}'")
        try:
            docs = self.vector_store.similarity_search(question, k=k)
            if not docs:
                return f"No relevant information found in the document for: '{question}'"

            # Format retrieved context
            context = "\n\n---\n\n".join([
                f"Source Chunk (Score: {doc.metadata.get('_score', 'N/A')}, Doc: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
                # FAISS often doesn't return score directly this way
                if hasattr(doc, 'metadata') and '_score' in doc.metadata
                else f"Source Chunk (Doc: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
                for doc in docs
            ])
            return f"Retrieved context for '{question}':\n\n{context}"
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return f"Error retrieving information for question: '{question}'"


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Ensure langchain/transformers are installed before running example
    if not LANGCHAIN_AVAILABLE or (TRANSFORMERS_AVAILABLE is False and HuggingFacePipeline is None):
        print("Required libraries not found. Please install dependencies before running the example.")
        print("pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers transformers torch accelerate bitsandbytes")
    else:
        # Import PDF extractor here, assuming it's in the right place
        try:
            from contract_to_text import PDFTextExtractor  # Adjust import path if necessary
        except ImportError:
            logger.error(
                "Could not import PDFTextExtractor. Ensure contract_to_text.py is accessible.")
            exit()

        # --- Configuration ---
        script_dir = Path(__file__).parent.resolve()
        # Use a smaller, well-supported model for easier testing, or specify your desired model
        # LLM_MODEL = "google/gemma-2-9b-it" # Example Gemma 2
        LLM_MODEL = "google/gemma-3-4b-it"  # Example Llama 3
        # LLM_MODEL = None # Set to None for retrieval-only mode

        # Define paths relative to the script location
        # Adjust these paths based on your project structure
        pdf_relative_path = "../data/CUAD_v1/full_contract_pdf/Part_I/License_Agreements/CytodynInc_20200109_10-Q_EX-10.5_11941634_EX-10.5_License Agreement.pdf"
        pdf_path = script_dir / pdf_relative_path
        output_dir = script_dir / "../output"
        vector_store_path = output_dir / "vector_stores"
        # --- End Configuration ---

        # 1. Extract text from PDF
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            logger.error(
                "Please ensure the data directory and PDF file exist relative to the script.")
        else:
            document_id = pdf_path.stem  # Use filename without extension as ID
            extractor = PDFTextExtractor()
            text = ""
            if extractor.load_pdf(pdf_path):
                text = extractor.extract_text()
                extractor.close()
            else:
                logger.error(f"Failed to extract text from {pdf_path}")

            if text:
                # 2. Initialize RAG system
                rag = TextChunkingRAG(
                    vector_store_dir=vector_store_path,
                    llm_model_name=LLM_MODEL
                )

                # 3. Process the document (creates/loads vector store)
                processing_results = rag.process_document(text, document_id)

                if processing_results:
                    logger.info(
                        f"Document Processing Summary: {processing_results}")

                    # 4. Ask a question
                    question = "What is the effective date of the License Agreement?"
                    answer = rag.answer_question(
                        question, k=3)  # Retrieve 3 chunks

                    print("\n" + "="*30)
                    print(f"Question: {question}")
                    print("="*30)
                    print("Answer/Retrieved Context:")
                    print(answer)
                    print("="*30 + "\n")

                    question_2 = "Who are the main parties involved in this agreement?"
                    answer_2 = rag.answer_question(question_2, k=4)

                    print("\n" + "="*30)
                    print(f"Question: {question_2}")
                    print("="*30)
                    print("Answer/Retrieved Context:")
                    print(answer_2)
                    print("="*30 + "\n")

                else:
                    logger.error(f"Failed to process document: {document_id}")
            else:
                logger.error(
                    f"No text extracted from {pdf_path}. Cannot proceed.")
# --- END OF FILE text_chunking_rag.py ---
