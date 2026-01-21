import pdfplumber

def extract_pdf_text_from_path(path):
    """
    Extract text from a PDF file with error handling.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        str: Extracted text from all pages
    """
    text = ""
    
    # FIX: Add try-except to handle corrupted or invalid PDFs
    try:
        with pdfplumber.open(path) as pdf:
            # FIX: Show progress for debugging
            print(f"📄 Extracting text from PDF: {path}")
            print(f"   Total pages: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                t = page.extract_text()
                if t:
                    text += t + "\n"
                    # Show progress every 10 pages
                    if page_num % 10 == 0:
                        print(f"   Processed {page_num}/{len(pdf.pages)} pages...")
            
            print(f"   ✓ Extraction complete. Total characters: {len(text)}")
    
    except Exception as e:
        # CRITICAL: Handle PDF extraction errors gracefully
        print(f"ERROR: Failed to extract text from PDF: {e}")
        raise ValueError(f"Could not read PDF file '{path}'. The file may be corrupted or password-protected.")
    
    # FIX: Validate that we extracted meaningful text
    if not text or len(text.strip()) < 10:
        # Warn if PDF appears to be empty or contains only whitespace
        print(f"WARNING: PDF appears to be empty or contains very little text ({len(text)} characters)")
        raise ValueError(f"PDF file '{path}' contains no extractable text. It may be an image-based PDF requiring OCR.")
    
    return text