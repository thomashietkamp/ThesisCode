from src.utils.text_chunking_rag import TextChunkingRAG
from src.utils.contract_to_text import PDFTextExtractor
import sys
import os
import argparse
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


def main():
    """
    Example script demonstrating how to use the TextChunkingRAG class for contract analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze a contract using RAG")
    parser.add_argument("pdf_path", help="Path to the PDF contract file")
    parser.add_argument("--output_dir", default="output",
                        help="Directory to save output files")
    parser.add_argument("--chunk_size", type=int,
                        default=1000, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int,
                        default=200, help="Overlap between chunks")
    parser.add_argument("--embeddings_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embeddings model to use")
    parser.add_argument("--llm_model", default=None,
                        help="HuggingFace language model to use (optional)")
    parser.add_argument("--query", default=None,
                        help="Question to ask about the contract (optional)")

    args = parser.parse_args()

    # Check if the PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return

    # Extract document ID from the PDF path
    pdf_filename = os.path.basename(args.pdf_path)
    document_id = os.path.splitext(pdf_filename)[0].replace(" ", "_")

    print(f"Processing contract: {args.pdf_path}")

    # Extract text from PDF
    extractor = PDFTextExtractor()
    if not extractor.load_pdf(args.pdf_path):
        print(f"Failed to load PDF file: {args.pdf_path}")
        return

    text = extractor.extract_text()
    extractor.close()

    # Save extracted text
    text_output_dir = os.path.join(args.output_dir, document_id)
    os.makedirs(text_output_dir, exist_ok=True)
    text_output_path = os.path.join(text_output_dir, "extracted_text.txt")

    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Extracted text saved to: {text_output_path}")

    # Process the text with RAG
    vector_store_path = os.path.join(args.output_dir, "vector_stores")

    rag = TextChunkingRAG(
        embeddings_model_name=args.embeddings_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=vector_store_path,
        llm_model_name=args.llm_model
    )

    # Process the document
    results = rag.process_document(text, document_id)

    # Add additional information to results
    results["document_id"] = document_id
    results["pdf_path"] = args.pdf_path
    results["extracted_text_path"] = text_output_path

    # Save the results
    rag.save_results(document_id, results, args.output_dir)

    # Print summary
    print("\nContract Analysis Summary:")
    print(f"Contract ID: {document_id}")
    print(f"Text Length: {results['text_length']} characters")
    print(f"Number of Chunks: {results['chunk_count']}")
    print(f"Vector Store: {results['vector_store']}")
    if 'vector_store_path' in results:
        print(f"Vector Store Path: {results['vector_store_path']}")

    # Answer a question if provided
    if args.query:
        print("\nQuery:", args.query)
        print("Relevant Sections:")
        answer = rag.answer_question(args.query)
        print(answer)
    else:
        # Default questions to demonstrate functionality
        default_questions = [
            "What are the payment terms in this contract?",
            "What are the termination clauses?",
            "Who are the parties involved in this agreement?",
            "What are the confidentiality provisions?"
        ]

        print("\nExample Query:")
        question = default_questions[0]
        print(f"Query: {question}")
        print("Relevant Sections:")
        answer = rag.answer_question(question)
        print(answer)


if __name__ == "__main__":
    main()
