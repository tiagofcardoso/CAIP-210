import fitz  # PyMuPDF
import os

pdf = fitz.open(r'c:\Users\tiago\OneDrive\CAIP-210\grok-CAIP-210.pdf')

# Create images folder
os.makedirs('pdf_images', exist_ok=True)

for page_num in range(len(pdf)):
    page = pdf[page_num]
    # Get high resolution image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pix.save(f'pdf_images/page_{page_num + 1:02d}.png')
    print(f"Saved page {page_num + 1}")

pdf.close()
print("Done! Images saved in pdf_images folder")
