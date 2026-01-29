"""
Enhanced PDF Text Extractor for CAIP-210 Exam Questions
Uses pdfplumber for better text extraction quality
"""

import os
import sys

def extract_pdf_with_pdfplumber(pdf_path, output_path=None):
    """
    Extract text from PDF using pdfplumber (better quality).
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_path (str): Path to the output text file (optional)
    
    Returns:
        str: Path to the created text file
    """
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        import pdfplumber
    
    try:
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = f"{base_name}_extracted_text.txt"
        
        # Open and read the PDF file
        extracted_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Processing {len(pdf.pages)} pages...")
            
            for page_num, page in enumerate(pdf.pages, 1):
                extracted_text += f"\n{'='*60}\n"
                extracted_text += f"PAGE {page_num}\n"
                extracted_text += f"{'='*60}\n\n"
                
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text
                else:
                    extracted_text += "[No text found on this page]\n"
                
                extracted_text += "\n"
                
                # Progress indicator
                if page_num % 10 == 0:
                    print(f"  Processed {page_num}/{len(pdf.pages)} pages...")
        
        # Write extracted text to output file
        with open(output_path, 'w', encoding='utf-8') as text_file:
            text_file.write(extracted_text)
        
        print(f"\n✅ Text successfully extracted from '{os.path.basename(pdf_path)}'")
        print(f"📄 Output saved to: '{output_path}'")
        print(f"📊 Total characters: {len(extracted_text):,}")
        return output_path
        
    except FileNotFoundError:
        print(f"❌ Error: PDF file '{pdf_path}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_multiple_pdfs(pdf_list):
    """Extract text from multiple PDFs."""
    results = {}
    for pdf_path in pdf_list:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(pdf_path)}")
        print(f"{'='*70}")
        result = extract_pdf_with_pdfplumber(pdf_path)
        results[pdf_path] = result
    return results

if __name__ == "__main__":
    # Define PDFs to extract
    pdfs_to_extract = [
        "AIP exam questions.pdf",
        "teste prep.pdf"
    ]
    
    print("🚀 CAIP-210 PDF Text Extractor")
    print("="*70)
    
    # Extract all PDFs
    results = extract_multiple_pdfs(pdfs_to_extract)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'='*70}")
    for pdf_path, output_path in results.items():
        status = "✅ Success" if output_path else "❌ Failed"
        print(f"{status}: {os.path.basename(pdf_path)}")
        if output_path:
            print(f"         → {output_path}")
    
    print("\n✅ All extractions completed!")
