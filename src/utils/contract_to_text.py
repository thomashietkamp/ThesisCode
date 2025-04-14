# --- START OF FILE contract_to_text.py ---

import os
from pathlib import Path
from typing import Optional, List
import logging

# Attempt to import PyPDF2 and handle potential ImportError
try:
    import PyPDF2
    from PyPDF2.errors import PdfReadError
except ImportError:
    print("PyPDF2 is not installed. Please install it: pip install pypdf2")
    PyPDF2 = None  # Set to None if import fails
    PdfReadError = Exception  # Use base Exception for catching errors

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """
    A class for extracting text from PDF files using PyPDF2.
    """

    def __init__(self, pdf_path: Optional[str | Path] = None):
        """
        Initialize the PDFTextExtractor with an optional PDF file path.

        Args:
            pdf_path (str | Path, optional): Path to the PDF file.
                                            Can be set later using load_pdf method.
        """
        if PyPDF2 is None:
            raise ImportError(
                "PyPDF2 is required but not installed. Please run: pip install pypdf2")

        self.pdf_path: Optional[Path] = Path(pdf_path) if pdf_path else None
        self.pdf_reader: Optional[PyPDF2.PdfReader] = None
        self.file_handle = None
        self.text_content: List[str] = []

        if self.pdf_path:
            self.load_pdf(self.pdf_path)

    def load_pdf(self, pdf_path: str | Path) -> bool:
        """
        Load a PDF file. Closes any previously opened file.

        Args:
            pdf_path (str | Path): Path to the PDF file.

        Returns:
            bool: True if the PDF was loaded successfully, False otherwise.
        """
        self.close()  # Ensure any previous file is closed
        self.pdf_path = Path(pdf_path)
        self.text_content = []  # Reset text content

        if not self.pdf_path.exists():
            logger.error(f"File not found at {self.pdf_path}")
            return False
        if not self.pdf_path.is_file():
            logger.error(f"Path is not a file: {self.pdf_path}")
            return False

        try:
            self.file_handle = open(self.pdf_path, 'rb')
            self.pdf_reader = PyPDF2.PdfReader(self.file_handle)
            # Check for encryption
            if self.pdf_reader.is_encrypted:
                logger.warning(
                    f"PDF is encrypted: {self.pdf_path}. Attempting to read anyway.")
                # You might need to handle decryption here if necessary, e.g., reader.decrypt('')
            logger.info(f"Successfully loaded PDF: {self.pdf_path}")
            return True
        except PdfReadError as e:
            logger.error(f"Error reading PDF file {self.pdf_path}: {e}")
            self.close()
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading PDF {self.pdf_path}: {e}")
            self.close()
            return False

    def extract_text(self) -> str:
        """
        Extract text from all pages of the loaded PDF.

        Returns:
            str: Extracted text concatenated with newline separators, or empty string if failed.
        """
        if not self.pdf_reader:
            logger.warning("No PDF loaded. Use load_pdf() method first.")
            return ""

        self.text_content = []
        num_pages = len(self.pdf_reader.pages)
        logger.info(f"Extracting text from {num_pages} pages...")

        for page_num in range(num_pages):
            try:
                page = self.pdf_reader.pages[page_num]
                page_text = page.extract_text()
                # Handle escaped unicode characters
                clean_text = page_text
                while '\\u' in clean_text:
                    try:
                        clean_text = clean_text.encode(
                            'utf-8').decode('unicode_escape')

                    except UnicodeDecodeError:
                        break

                if clean_text:
                    # Greek question mark (;), if it appears as punctuation
                    clean_text = clean_text.replace(
                        "\u00a0", " ")
                    clean_text = clean_text.replace("\u037e", ";")
                    clean_text = clean_text.replace(
                        "\u201c", '"').replace("\u201d", '"')

                    clean_text = clean_text.strip()  # Remove leading/trailing whitespace

                    self.text_content.append(clean_text)
                else:
                    logger.warning(
                        f"Could not extract text from page {page_num} of {self.pdf_path}")
            except Exception as e:
                logger.error(
                    f"Error extracting text from page {page_num} of {self.pdf_path}: {e}")
                # Continue to next page

        extracted_text = "\n\n".join(self.text_content)
        logger.info(f"Extracted {len(extracted_text)} characters.")
        return extracted_text

    def extract_text_from_page(self, page_num: int) -> str:
        """
        Extract text from a specific page of the loaded PDF.

        Args:
            page_num (int): Page number (0-indexed).

        Returns:
            str: Extracted text from the specified page, or empty string if failed.
        """
        if not self.pdf_reader:
            logger.warning("No PDF loaded. Use load_pdf() method first.")
            return ""

        num_pages = len(self.pdf_reader.pages)
        if not 0 <= page_num < num_pages:
            logger.error(
                f"Page number {page_num} is out of range (0-{num_pages - 1}).")
            return ""

        try:
            page = self.pdf_reader.pages[page_num]
            page_text = page.extract_text()
            return page_text if page_text else ""
        except Exception as e:
            logger.error(
                f"Error extracting text from page {page_num} of {self.pdf_path}: {e}")
            return ""

    def get_page_count(self) -> int:
        """
        Get the number of pages in the loaded PDF.

        Returns:
            int: Number of pages, or 0 if no PDF is loaded.
        """
        return len(self.pdf_reader.pages) if self.pdf_reader else 0

    def save_text_to_file(self, output_path: str | Path) -> bool:
        """
        Save the extracted text (from all pages) to a file.

        Args:
            output_path (str | Path): Path to save the text file.

        Returns:
            bool: True if the text was saved successfully, False otherwise.
        """
        if not self.text_content:
            text_to_save = self.extract_text()  # Ensure text is extracted if not already
            if not text_to_save:
                logger.warning(
                    "No text content extracted or available to save.")
                return False
        else:
            text_to_save = "\n\n".join(self.text_content)

        output_path = Path(output_path)
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(text_to_save)
            logger.info(f"Extracted text saved successfully to {output_path}")
            return True
        except IOError as e:
            logger.error(f"Error saving text to file {output_path}: {e}")
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred saving text to {output_path}: {e}")
            return False

    def close(self):
        """
        Close the PDF file handle if it's open.
        """
        if self.file_handle:
            try:
                self.file_handle.close()
                logger.debug(f"Closed file handle for {self.pdf_path}")
            except Exception as e:
                logger.error(
                    f"Error closing file handle for {self.pdf_path}: {e}")
            finally:
                self.file_handle = None
                self.pdf_reader = None
                # Keep self.pdf_path, it might be useful

    def __del__(self):
        """
        Ensure file is closed when object is garbage collected.
        """
        self.close()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Define paths (use relative paths carefully or absolute paths)
    # Assuming 'data' and 'output' directories exist relative to the script location
    script_dir = Path(__file__).parent.resolve()
    pdf_path = script_dir.parent / \
        "data/CUAD_v1/full_contract_pdf/Part_I/License_Agreements/CytodynInc_20200109_10-Q_EX-10.5_11941634_EX-10.5_License Agreement.pdf"
    output_txt_path = script_dir.parent / "output/extracted_text.txt"

    # Check if PDF exists before proceeding
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        logger.error("Please ensure the data directory and PDF file exist.")
    else:
        # Create an instance of PDFTextExtractor
        extractor = PDFTextExtractor()

        # Load a PDF file
        if extractor.load_pdf(pdf_path):
            # Extract text from all pages
            text = extractor.extract_text()
            # print(text[:1000] + "...") # Print first 1000 chars for brevity

            # Save extracted text to a file
            if extractor.save_text_to_file(output_txt_path):
                logger.info(f"Text saved to {output_txt_path}")
            else:
                logger.error("Failed to save extracted text.")

            # Close the file explicitly (though __del__ handles it too)
            extractor.close()
        else:
            logger.error(f"Failed to load PDF: {pdf_path}")

# --- END OF FILE contract_to_text.py ---
