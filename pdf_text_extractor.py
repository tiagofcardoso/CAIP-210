import PyPDF2
import os
import sys

def extract_text_from_pdf(pdf_path, output_path=None):
    """
    Extract text from a PDF file and save it to a text file.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_path (str): Path to the output text file (optional)
    
    Returns:
        str: Path to the created text file
    """
    try:
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = f"{base_name}_extracted_text.txt"
        
        # Open and read the PDF file
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            extracted_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += f"\n--- Page {page_num + 1} ---\n"
                extracted_text += page.extract_text()
                extracted_text += "\n"
        
        # Write extracted text to output file
        with open(output_path, 'w', encoding='utf-8') as text_file:
            text_file.write(extracted_text)
        
        print(f"Text successfully extracted from '{pdf_path}'")
        print(f"Output saved to: '{output_path}'")
        return output_path
        
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
        return None
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

def main():
    """Main function to handle command line arguments or interactive input."""
    if len(sys.argv) > 1:
        # Use command line argument
        pdf_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Interactive input
        pdf_path = input("Enter the path to the PDF file: ").strip()
        output_path = input("Enter output text file path (or press Enter for auto-naming): ").strip()
        if not output_path:
            output_path = None
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        return
    
    # Extract text from PDF
    result = extract_text_from_pdf(pdf_path, output_path)
    
    if result:
        print("Text extraction completed successfully!")
    else:
        print("Text extraction failed.")

if __name__ == "__main__":
    main()
