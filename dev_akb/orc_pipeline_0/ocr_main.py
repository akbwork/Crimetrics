import os
import json
from pdf_converter import convert_pdf_to_png
from ocr_extractor import OcrExtractor
from ollama_parsing_agent import OllamaParsingAgent
from time_it import time_it

@time_it
def process_fir(pdf_path, output_dir=None):
    """
    Process an FIR PDF document into structured JSON.
    
    Args:
        pdf_path (str): Path to the FIR PDF file
        output_dir (str, optional): Directory to save the JSON output
        
    Returns:
        dict: Extracted FIR data
    """
    # Set default output directory if not provided
    if output_dir is None:
        # Create output directory relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '../output/json')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the PDF basename for naming consistency
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    print(f"Processing FIR: {pdf_basename}")
    
    # Step 1: Convert PDF to PNG
    print("Step 1: Converting PDF to PNG...")
    png_paths = convert_pdf_to_png(pdf_path)
    print(f"Created {len(png_paths)} PNG files.")
    
    # Step 2: Extract text using OCR
    print("Step 2: Performing OCR...")
    ocr_text = OcrExtractor.get_combined_text_from_pdf(pdf_basename)
    
    # Print a sample of the OCR text
    print(f"OCR Text Sample: {ocr_text[:200]}...")
    
    # Step 3: Parse text into structured JSON
    print("Step 3: Parsing text into structured data...")
    agent = OllamaParsingAgent(model="llama3")
    fir_data = agent.parse_fir(ocr_text)
    
    # Save the extracted data to JSON file
    output_path = os.path.join(output_dir, f"{pdf_basename}_data.json")
    
    # FIXED LINE BELOW - Added with block and file object
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fir_data, f, indent=2, ensure_ascii=False)
    
    print(f"FIR data saved to: {output_path}")
    
    # Print summary of extracted data
    print("\nExtracted FIR Information:")
    print(f"FIR Number: {fir_data.get('FIR', {}).get('FIR_No', 'Not found')}")
    print(f"Police Station: {fir_data.get('FIR', {}).get('Police_Station', 'Not found')}")
    print(f"Complainant Name: {fir_data.get('Complainant_Informant', {}).get('Name', 'Not found')}")
    
    return fir_data

if __name__ == "__main__":
    # Example usage
    pdf_path = "/Users/ananthakrishnab/Desktop/Projects/Crimmetrics_Abi/Crimetrics/dev_akb/data/FIR_DEMO.pdf"
    fir_data = process_fir(pdf_path)