from src.utils.contract_to_text import PDFTextExtractor
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


def main():
    """
    Example script demonstrating how to use the PDFTextExtractor class.
    """
    # Check if a PDF file path was provided as a command-line argument
    if len(sys.argv) < 2:
        print(
            "Usage: python pdf_extraction_example.py <path_to_pdf_file> [output_text_file]")
        return

    # Get the PDF file path from the command-line arguments
    pdf_path = sys.argv[1]

    # Create an instance of PDFTextExtractor
    extractor = PDFTextExtractor()

    # Load the PDF file
    if not extractor.load_pdf(pdf_path):
        print(f"Failed to load PDF file: {pdf_path}")
        return

    # Print information about the PDF
    page_count = extractor.get_page_count()
    print(f"PDF loaded successfully. Number of pages: {page_count}")

    # Extract text from all pages
    print("Extracting text from all pages...")
    text = extractor.extract_text()

    # Print a preview of the extracted text (first 500 characters)
    preview_length = min(500, len(text))
    print(f"\nText preview (first {preview_length} characters):")
    print("-" * 50)
    print(text[:preview_length] + "..." if len(text)
          > preview_length else text)
    print("-" * 50)

    # Save the extracted text to a file if an output file path was provided
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
        print(f"\nSaving extracted text to: {output_path}")
        if extractor.save_text_to_file(output_path):
            print("Text saved successfully.")
        else:
            print("Failed to save text to file.")

    print("\nDone!")


if __name__ == "__main__":
    main()
