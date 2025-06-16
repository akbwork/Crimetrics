import streamlit as st

# This must be the first Streamlit command
st.set_page_config(
    page_title="FIR Digitisation | Crimmetrics",
    page_icon="📄",
    layout="wide"
)

import os
import json
import tempfile
import pandas as pd
import sys
import re

# Add the OCR pipeline directory to Python path to import modules
# Get the absolute path to the orc_pipeline_0 directory
orc_pipeline_path = "/Users/ananthakrishnab/Desktop/Projects/Crimmetrics_Abi/Crimetrics/dev_akb/orc_pipeline_0"
if orc_pipeline_path not in sys.path:
    sys.path.append(orc_pipeline_path)

# Now import the modules from the orc_pipeline_0 directory
try:
    # Import OCR pipeline modules directly
    from pdf_converter import convert_pdf_to_png
    from ocr_extractor import OcrExtractor
    from ollama_parsing_agent import OllamaParsingAgent
    from time_it import time_it
    
    st.success("Created AI Extraction Pipeline")
except ImportError as e:
    st.error(f"Error importing OCR pipeline modules: {e}")
    st.info("Please make sure the OCR pipeline modules are in the correct location")
    # Create placeholder functions for demonstration
    def convert_pdf_to_png(pdf_path, output_dir=None, dpi=300):
        st.warning("Using placeholder PDF converter (module not loaded)")
        return [pdf_path]
    
    class OcrExtractor:
        @staticmethod
        def get_combined_text_from_pdf(pdf_basename, png_dir=None):
            st.warning("Using placeholder OCR extractor (module not loaded)")
            return "Sample OCR text - module not properly loaded"
    
    class OllamaParsingAgent:
        def __init__(self, model="deepseek-r1:latest"):
            self.model = model
        
        def parse_fir(self, ocr_text):
            st.warning("Using placeholder parsing agent (module not loaded)")
            return self._empty_fir_template()
        
        def _empty_fir_template(self):
            return {
                "FIR": {"District": "", "Year": "", "FIR_No": "", "Date": "", "Police_Station": "", "Section": "",
                        "Occurrence_of_Offense": {"Day": "", "Date": "", "Time_From": ""},
                        "Information_Received_At_PS": {"Date": "", "Time": ""},
                        "General_Diary_Reference": {"Entry_No": "", "Date": "", "Time": ""},
                        "Type_of_Information": "",
                        "Place_of_Occurrence": {"Direction_and_Distance_From_PS": "", "Beat_No": "", "Place": "",
                                              "Street_Village": "", "Area_Mandal": "", "City_District": "", "State": ""}},
                "Complainant_Informant": {"Name": "", "Father_Husband_Name": "", "Age": "", "Nationality": "", "Occupation": "",
                                        "Address": {"Flat_No": "", "Street_Village": "", "Area_Mandal": "", "City_District": "", "State": ""},
                                        "Phone_No": ""},
                "Accused": {"Serial_No": "", "Name": "", "Cab_No": "", "Nationality": "", "Management": ""},
                "Victim": {"Name": "", "Father_Name": "", "Age": "", "Occupation": "",
                         "Address": {"Flat_No": "", "Street_Village": "", "Area_Mandal": "", "City_District": ""},
                         "Injuries": "", "Condition": "", "Treatment": {"Initial": "", "Shifted_To": ""}},
                "Incident_Details": {"Date": "", "Time": "", "Location": "", "Description": ""},
                "Investigation": {"Case_Registered_U_S": "", "Case_No": "",
                                "Investigation_Officer": {"Rank": "", "Name": ""},
                                "Officer_in_Charge": {"Name": "", "Rank": "", "Signature": ""},
                                "Date_Dispatch_Court": ""},
                "Reasons_for_Delay": "", "Court": ""
            }
    
    def time_it(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

st.title("FIR Digitisation Module")
st.write("Upload an FIR document for automatic digitisation and structured data extraction.")

def extract_basic_info_from_ocr(ocr_text, fir_data):
    """Extract basic information from OCR text to help fill in gaps"""
    # FIR Number
    fir_match = re.search(r'FIR No\.?\s*[:.]?\s*([0-9\/]+)', ocr_text, re.IGNORECASE)
    if fir_match and not fir_data["FIR"]["FIR_No"]:
        fir_data["FIR"]["FIR_No"] = fir_match.group(1).strip()
        
        # Try to extract year from FIR number
        year_match = re.search(r'/(\d{4})', fir_match.group(1).strip())
        if year_match and not fir_data["FIR"]["Year"]:
            fir_data["FIR"]["Year"] = year_match.group(1)
    
    # Police Station
    ps_match = re.search(r'Police Station\s*[:.]?\s*([A-Za-z\s]+)', ocr_text, re.IGNORECASE)
    if ps_match and not fir_data["FIR"]["Police_Station"]:
        fir_data["FIR"]["Police_Station"] = ps_match.group(1).strip()
    
    # Section
    section_match = re.search(r'section\s*[:.]?\s*([0-9\s,]+\s*[A-Za-z\s]+)', ocr_text, re.IGNORECASE)
    if section_match and not fir_data["FIR"]["Section"]:
        fir_data["FIR"]["Section"] = section_match.group(1).strip()
    
    # Complainant Name
    name_match = re.search(r'complainant[\'s]*\s*name\s*[:.]?\s*([A-Za-z\.\s]+)', ocr_text, re.IGNORECASE)
    if name_match and not fir_data["Complainant_Informant"]["Name"]:
        fir_data["Complainant_Informant"]["Name"] = name_match.group(1).strip()
    
    # Add this section to extract incident description
    # Look for common patterns that might indicate an incident description
    description_patterns = [
        r'(?:brief\s+facts|incident\s+details|description\s+of\s+offence|brief\s+description).*?[:.]\s*(.*?)(?=\n\n|\Z)',
        r'(?:statement\s+of\s+complainant|facts\s+of\s+the\s+case).*?[:.]\s*(.*?)(?=\n\n|\Z)',
        r'(?:On\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}.*?at\s+about.*?hrs)(.{50,}?)(?=\n\n|\Z)'
    ]
    
    # Try each pattern to find a description
    for pattern in description_patterns:
        desc_match = re.search(pattern, ocr_text, re.IGNORECASE | re.DOTALL)
        if desc_match and not fir_data["Incident_Details"]["Description"]:
            # Clean up the description (remove extra spaces, newlines)
            description = desc_match.group(1).strip()
            description = re.sub(r'\s+', ' ', description)
            
            # If description is too short, it might be incomplete
            if len(description) > 50:  # Only use if reasonably long
                fir_data["Incident_Details"]["Description"] = description
                break
    
    # If no specific pattern matched, try to extract a large text block that might be the description
    if not fir_data["Incident_Details"]["Description"]:
        # Look for a paragraph after the complainant info and before the signature/officer info
        large_text_blocks = re.findall(r'\n\n(.{100,}?)\n\n', ocr_text, re.DOTALL)
        if large_text_blocks:
            # Get the longest text block, it's likely the description
            longest_block = max(large_text_blocks, key=len)
            fir_data["Incident_Details"]["Description"] = re.sub(r'\s+', ' ', longest_block.strip())
    
    return fir_data

def display_fir_data_tables(fir_data):
    """Display FIR data in structured tables"""
    
    # 1. FIR Basic Information
    with st.expander("FIR Basic Information", expanded=True):
        # Main FIR details
        fir_info = fir_data.get("FIR", {})
        fir_basic_df = pd.DataFrame({
            'Field': ['District', 'Year', 'FIR_No', 'Date', 'Police_Station', 'Section', 'Type_of_Information'],
            'Value': [
                fir_info.get('District', ''),
                fir_info.get('Year', ''),
                fir_info.get('FIR_No', ''),
                fir_info.get('Date', ''),
                fir_info.get('Police_Station', ''),
                fir_info.get('Section', ''),
                fir_info.get('Type_of_Information', '')
            ]
        })
        st.table(fir_basic_df)
        
        # Occurrence of Offense
        st.subheader("Occurrence of Offense")
        occurrence = fir_info.get('Occurrence_of_Offense', {})
        occurrence_df = pd.DataFrame({
            'Field': ['Day', 'Date', 'Time_From'],
            'Value': [
                occurrence.get('Day', ''),
                occurrence.get('Date', ''),
                occurrence.get('Time_From', '')
            ]
        })
        st.table(occurrence_df)
        
        # Information Received at PS
        st.subheader("Information Received at PS")
        info_received = fir_info.get('Information_Received_At_PS', {})
        info_received_df = pd.DataFrame({
            'Field': ['Date', 'Time'],
            'Value': [
                info_received.get('Date', ''),
                info_received.get('Time', '')
            ]
        })
        st.table(info_received_df)
        
        # General Diary Reference
        st.subheader("General Diary Reference")
        diary_ref = fir_info.get('General_Diary_Reference', {})
        diary_ref_df = pd.DataFrame({
            'Field': ['Entry_No', 'Date', 'Time'],
            'Value': [
                diary_ref.get('Entry_No', ''),
                diary_ref.get('Date', ''),
                diary_ref.get('Time', '')
            ]
        })
        st.table(diary_ref_df)
        
        # Place of Occurrence
        st.subheader("Place of Occurrence")
        place = fir_info.get('Place_of_Occurrence', {})
        place_df = pd.DataFrame({
            'Field': ['Direction_and_Distance_From_PS', 'Beat_No', 'Place', 
                     'Street_Village', 'Area_Mandal', 'City_District', 'State'],
            'Value': [
                place.get('Direction_and_Distance_From_PS', ''),
                place.get('Beat_No', ''),
                place.get('Place', ''),
                place.get('Street_Village', ''),
                place.get('Area_Mandal', ''),
                place.get('City_District', ''),
                place.get('State', '')
            ]
        })
        st.table(place_df)
    
    # 2. Complainant/Informant Details
    with st.expander("Complainant/Informant Details", expanded=False):
        # Basic complainant details
        complainant = fir_data.get('Complainant_Informant', {})
        complainant_df = pd.DataFrame({
            'Field': ['Name', 'Father_Husband_Name', 'Age', 'Nationality', 'Occupation', 'Phone_No'],
            'Value': [
                complainant.get('Name', ''),
                complainant.get('Father_Husband_Name', ''),
                complainant.get('Age', ''),
                complainant.get('Nationality', ''),
                complainant.get('Occupation', ''),
                complainant.get('Phone_No', '')
            ]
        })
        st.table(complainant_df)
        
        # Complainant Address
        st.subheader("Complainant Address")
        comp_addr = complainant.get('Address', {})
        comp_addr_df = pd.DataFrame({
            'Field': ['Flat_No', 'Street_Village', 'Area_Mandal', 'City_District', 'State'],
            'Value': [
                comp_addr.get('Flat_No', ''),
                comp_addr.get('Street_Village', ''),
                comp_addr.get('Area_Mandal', ''),
                comp_addr.get('City_District', ''),
                comp_addr.get('State', '')
            ]
        })
        st.table(comp_addr_df)
    
    # 3. Accused Details
    with st.expander("Accused Details", expanded=False):
        accused = fir_data.get('Accused', {})
        accused_df = pd.DataFrame({
            'Field': ['Serial_No', 'Name', 'Cab_No', 'Nationality', 'Management'],
            'Value': [
                accused.get('Serial_No', ''),
                accused.get('Name', ''),
                accused.get('Cab_No', ''),
                accused.get('Nationality', ''),
                accused.get('Management', '')
            ]
        })
        st.table(accused_df)
    
    # 4. Victim Details
    with st.expander("Victim Details", expanded=False):
        # Basic victim details
        victim = fir_data.get('Victim', {})
        victim_df = pd.DataFrame({
            'Field': ['Name', 'Father_Name', 'Age', 'Occupation', 'Injuries', 'Condition'],
            'Value': [
                victim.get('Name', ''),
                victim.get('Father_Name', ''),
                victim.get('Age', ''),
                victim.get('Occupation', ''),
                victim.get('Injuries', ''),
                victim.get('Condition', '')
            ]
        })
        st.table(victim_df)
        
        # Victim Address
        st.subheader("Victim Address")
        victim_addr = victim.get('Address', {})
        victim_addr_df = pd.DataFrame({
            'Field': ['Flat_No', 'Street_Village', 'Area_Mandal', 'City_District'],
            'Value': [
                victim_addr.get('Flat_No', ''),
                victim_addr.get('Street_Village', ''),
                victim_addr.get('Area_Mandal', ''),
                victim_addr.get('City_District', '')
            ]
        })
        st.table(victim_addr_df)
        
        # Victim Treatment
        st.subheader("Victim Treatment")
        treatment = victim.get('Treatment', {})
        treatment_df = pd.DataFrame({
            'Field': ['Initial', 'Shifted_To'],
            'Value': [
                treatment.get('Initial', ''),
                treatment.get('Shifted_To', '')
            ]
        })
        st.table(treatment_df)
    
    # 5. Incident Details
    with st.expander("Incident Details", expanded=False):
        incident = fir_data.get('Incident_Details', {})
        
        # Display date, time, location in a table
        incident_basic_df = pd.DataFrame({
            'Field': ['Date', 'Time', 'Location'],
            'Value': [
                incident.get('Date', ''),
                incident.get('Time', ''),
                incident.get('Location', '')
            ]
        })
        st.table(incident_basic_df)
        
        # Display the description in a dedicated text area with more space
        st.subheader("Incident Description")
        description = incident.get('Description', '')
        if description:
            st.text_area("", description, height=150, disabled=True)
        else:
            st.info("No incident description available")
    
    # 6. Investigation Details
    with st.expander("Investigation Details", expanded=False):
        # Basic investigation details
        investigation = fir_data.get('Investigation', {})
        investigation_df = pd.DataFrame({
            'Field': ['Case_Registered_U_S', 'Case_No', 'Date_Dispatch_Court'],
            'Value': [
                investigation.get('Case_Registered_U_S', ''),
                investigation.get('Case_No', ''),
                investigation.get('Date_Dispatch_Court', '')
            ]
        })
        st.table(investigation_df)
        
        # Investigation Officer
        st.subheader("Investigation Officer")
        io = investigation.get('Investigation_Officer', {})
        io_df = pd.DataFrame({
            'Field': ['Rank', 'Name'],
            'Value': [
                io.get('Rank', ''),
                io.get('Name', '')
            ]
        })
        st.table(io_df)
        
        # Officer in Charge
        st.subheader("Officer in Charge")
        oic = investigation.get('Officer_in_Charge', {})
        oic_df = pd.DataFrame({
            'Field': ['Name', 'Rank', 'Signature'],
            'Value': [
                oic.get('Name', ''),
                oic.get('Rank', ''),
                oic.get('Signature', '')
            ]
        })
        st.table(oic_df)
    
    # 7. Additional Information
    with st.expander("Additional Information", expanded=False):
        additional_df = pd.DataFrame({
            'Field': ['Reasons_for_Delay', 'Court'],
            'Value': [
                fir_data.get('Reasons_for_Delay', ''),
                fir_data.get('Court', '')
            ]
        })
        st.table(additional_df)

@time_it
def process_fir_streamlit(pdf_path, temp_dir):
    """
    Process an FIR PDF document for Streamlit display.
    
    Args:
        pdf_path (str): Path to the FIR PDF file
        temp_dir (str): Temporary directory for processing
        
    Returns:
        dict: Extracted FIR data
    """
    # Create PNG output directory in temp directory
    png_dir = os.path.join(temp_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    
    # Get the PDF basename for naming consistency
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Step 1: Convert PDF to PNG
    progress_text = st.empty()
    progress_text.text("Step 1/3: Converting PDF to PNG...")
    png_paths = convert_pdf_to_png(pdf_path, output_dir=png_dir)
    progress_text.text(f"Step 1/3 Complete: Created {len(png_paths)} PNG files.")
    
    # Step 2: Extract text using OCR
    progress_text.text("Step 2/3: Reading PDF (this may take a minute)...")
    ocr_text = OcrExtractor.get_combined_text_from_pdf(pdf_basename, png_dir=png_dir)
    progress_text.text("Step 2/3 Complete: PDF Reading.")
    
    # Step 3: Parse text into structured JSON
    progress_text.text("Step 3/3: Processing Text...")
    
    try:
        agent = OllamaParsingAgent(model="llama3")
        fir_data = agent.parse_fir(ocr_text)
        
        # Make sure to call extract_basic_info_from_ocr
        fir_data = extract_basic_info_from_ocr(ocr_text, fir_data)
        
        progress_text.text("Processing complete! Displaying results...")
    except Exception as e:
        st.warning(f"Error connecting to Ollama: {e}")
        st.info("Using fallback extraction method.")
        # Return empty template if Ollama fails
        agent = OllamaParsingAgent()
        fir_data = agent._empty_fir_template()
        
        # Even with fallback, try to extract info from OCR
        fir_data = extract_basic_info_from_ocr(ocr_text, fir_data)
    
    return fir_data

# Main app flow
uploaded_file = st.file_uploader("Upload FIR PDF", type=["pdf"])

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to the temporary directory
        temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process button
        if st.button("Process FIR Document"):
            try:
                with st.spinner("Processing document..."):
                    # Process the FIR document
                    fir_data = process_fir_streamlit(temp_pdf_path, temp_dir)
                
                # Display the FIR data in tables
                display_fir_data_tables(fir_data)
                
                # Add download button for JSON
                json_str = json.dumps(fir_data, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=json_str,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_data.json",
                    mime="application/json"
                )
                
                # Add edit and save buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Edit Details"):
                        st.info("Edit functionality to be implemented.")
                with col2:
                    if st.button("Save to Database"):
                        st.info("Save to database functionality to be implemented.")
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

# '''

# # Add a note about supported formats
# st.markdown("""
# ---
# **Note:** This module currently supports PDF files. For best results, upload clear scans of FIR documents.

# **Process Flow:**
# 1. Upload FIR PDF document
# 2. Document is converted to PNG images
# 3. Text is extracted using OCR
# 4. Structured information is parsed using NLP
# 5. Results are displayed in organized tables
# """)
