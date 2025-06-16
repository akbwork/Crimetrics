import os
from paddleocr import PaddleOCR
from time_it import time_it

# Define configurable output directories
OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__), 'output')
PNG_DIR = os.path.join(OUTPUT_BASE_DIR, 'png')

class OcrExtractor:
    """Simple OCR extractor using PaddleOCR."""
    
    @time_it
    @staticmethod
    def ocr_paddle(
        png_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    ):
        """
        Runs OCR on the given image and returns results as a list of text.
        
        Args:
            png_path (str): Path to the PNG image file
            use_doc_orientation_classify (bool): PaddleOCR param
            use_doc_unwarping (bool): PaddleOCR param
            use_textline_orientation (bool): PaddleOCR param
            
        Returns:
            list: OCR extracted text list
        """
        ocr = PaddleOCR(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation
        )
        
        result = ocr.predict(input=png_path)
        if result and len(result) > 0:
            return result[0]["rec_texts"]
        return []

    @time_it
    @staticmethod
    def ocr_paddle_multi_page(
        pdf_basename,
        png_dir=PNG_DIR,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    ):
        """
        Runs OCR on all PNG files related to a specific PDF and combines results.
        
        Args:
            pdf_basename (str): Base name of the PDF file without extension
            png_dir (str): Directory containing PNG files
            use_doc_orientation_classify (bool): PaddleOCR param
            use_doc_unwarping (bool): PaddleOCR param
            use_textline_orientation (bool): PaddleOCR param
            
        Returns:
            dict: Combined OCR results with page numbers as keys
        """
        ocr = PaddleOCR(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation
        )
        
        # Find all PNG files related to this PDF
        related_pngs = []
        for filename in os.listdir(png_dir):
            if filename.endswith('.png') and pdf_basename in filename:
                try:
                    page_num = int(filename.split('page_')[-1].split('.')[0])
                    related_pngs.append((page_num, os.path.join(png_dir, filename)))
                except (ValueError, IndexError):
                    related_pngs.append((999, os.path.join(png_dir, filename)))
        
        # Sort by page number
        related_pngs.sort(key=lambda x: x[0])
        
        if not related_pngs:
            print(f"No PNG files found for PDF '{pdf_basename}' in directory {png_dir}")
            return {}
        
        # Process each PNG and combine results
        combined_results = {}
        for page_num, png_path in related_pngs:
            print(f"Processing page {page_num}: {os.path.basename(png_path)}")
            result = ocr.predict(input=png_path)
            
            if result and len(result) > 0:
                rec_texts = result[0]["rec_texts"]
                
                page_key = f"page_{page_num}" if page_num != 999 else f"page_unknown_{len(combined_results)}"
                combined_results[page_key] = rec_texts
        
        return combined_results
    
    @time_it
    @staticmethod
    def get_combined_text_from_pdf(pdf_basename, png_dir=PNG_DIR):
        """
        Get all text from a multi-page PDF as a single string.
        
        Args:
            pdf_basename (str): Base name of the PDF file without extension
            png_dir (str): Directory containing PNG files
            
        Returns:
            str: Combined text from all pages
        """
        combined_results = OcrExtractor.ocr_paddle_multi_page(
            pdf_basename=pdf_basename,
            png_dir=png_dir
        )
        
        all_text = []
        for page_key in sorted(combined_results.keys()):
            page_text = combined_results[page_key]
            if isinstance(page_text, list):
                all_text.extend(page_text)
            else:
                all_text.append(str(page_text))
                
        return "\n".join(all_text)

if __name__ == "__main__":
    # Example usage
    response = OcrExtractor.ocr_paddle("output/png/sample_fir_page_1.png")
    print(f"OCR Result: {response[:100]}...")
    
    # Multi-page processing example
    all_text = OcrExtractor.get_combined_text_from_pdf("sample_fir")
    print(f"Combined text: {all_text[:200]}...")