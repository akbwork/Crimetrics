import os
from pdf2image import convert_from_path
from time_it import time_it

# Define configurable output directories
OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__), 'output')
PNG_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'png')

@time_it
def convert_pdf_to_png(pdf_path, output_dir=PNG_OUTPUT_DIR, dpi=300):
    """
    Convert a PDF file to PNG images and save them in the specified directory.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save the PNG files
        dpi (int, optional): DPI resolution for the output images

    Returns:
        list: Paths to the saved PNG files
    """
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=dpi)
    png_paths = []
    
    for i, image in enumerate(images):
        png_path = os.path.join(output_dir, f"{pdf_filename}_page_{i+1}.png")
        image.save(png_path, "PNG")
        png_paths.append(png_path)
        
    return png_paths

if __name__ == "__main__":
    # Example usage
    saved_png_paths = convert_pdf_to_png("pdfs/sample_fir.pdf")
    print(f"Saved PNGs: {saved_png_paths}")