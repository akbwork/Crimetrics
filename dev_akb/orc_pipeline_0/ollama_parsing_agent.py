import json
import requests
from typing import Dict, List, Union
from time_it import time_it

class OllamaParsingAgent:
    """Uses Ollama to parse and structure OCR text from FIR documents."""

    def __init__(self, model="llama3", api_base="http://localhost:11434"):
        """
        Initialize the Ollama parsing agent.
        
        Args:
            model (str): Name of the Ollama model to use
            api_base (str): Base URL for Ollama API
        """
        self.model = model
        self.api_base = api_base

    @time_it
    def parse_fir(self, ocr_text: Union[List[str], str]) -> Dict:
        """
        Parse FIR OCR text using Ollama and extract structured FIR information.
        
        Args:
            ocr_text: OCR text as list of strings or single string
            
        Returns:
            Dict: Structured information extracted from the FIR
        """
        # Convert list to string if needed
        if isinstance(ocr_text, list):
            text = "\n".join(ocr_text)
        else:
            text = ocr_text

        # Define the prompt for Ollama
        prompt = f"""
You are an expert legal document parser. Extract the following FIR fields from the OCR text below. 
If a field is not found, use an empty string or "NA". Return a valid JSON object matching the structure and field names below. 
Do not include explanations, only the JSON.

FIR JSON STRUCTURE:
{{
  "FIR": {{
    "District": "",
    "Year": "",
    "FIR_No": "",
    "Date": "",
    "Police_Station": "",
    "Section": "",
    "Occurrence_of_Offense": {{
      "Day": "",
      "Date": "",
      "Time_From": ""
    }},
    "Information_Received_At_PS": {{
      "Date": "",
      "Time": ""
    }},
    "General_Diary_Reference": {{
      "Entry_No": "",
      "Date": "",
      "Time": ""
    }},
    "Type_of_Information": "",
    "Place_of_Occurrence": {{
      "Direction_and_Distance_From_PS": "",
      "Beat_No": "",
      "Place": "",
      "Street_Village": "",
      "Area_Mandal": "",
      "City_District": "",
      "State": ""
    }}
  }},
  "Complainant_Informant": {{
    "Name": "",
    "Father_Husband_Name": "",
    "Age": "",
    "Nationality": "",
    "Occupation": "",
    "Address": {{
      "Flat_No": "",
      "Street_Village": "",
      "Area_Mandal": "",
      "City_District": "",
      "State": ""
    }},
    "Phone_No": ""
  }},
  "Accused": {{
    "Serial_No": "",
    "Name": "",
    "Cab_No": "",
    "Nationality": "",
    "Management": ""
  }},
  "Victim": {{
    "Name": "",
    "Father_Name": "",
    "Age": "",
    "Occupation": "",
    "Address": {{
      "Flat_No": "",
      "Street_Village": "",
      "Area_Mandal": "",
      "City_District": ""
    }},
    "Injuries": "",
    "Condition": "",
    "Treatment": {{
      "Initial": "",
      "Shifted_To": ""
    }}
  }},
  "Incident_Details": {{
    "Date": "",
    "Time": "",
    "Location": "",
    "Description": ""
  }},
  "Investigation": {{
    "Case_Registered_U_S": "",
    "Case_No": "",
    "Investigation_Officer": {{
      "Rank": "",
      "Name": ""
    }},
    "Officer_in_Charge": {{
      "Name": "",
      "Rank": "",
      "Signature": ""
    }},
    "Date_Dispatch_Court": ""
  }},
  "Reasons_for_Delay": "",
  "Court": ""
}}

OCR TEXT:
{text}

Return ONLY the JSON object above, filled with extracted values.
"""

        # Call Ollama API
        response = self._call_ollama(prompt)
        
        # Parse and validate the response
        try:
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"Error parsing Ollama response: {e}")
            print(f"Raw response: {response}")
            # Return empty template as fallback
            return self._empty_fir_template()

    def _call_ollama(self, prompt: str) -> str:
        """
        Call the Ollama API with the given prompt.
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            str: The response text from Ollama
        """
        try:
            url = f"{self.api_base}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return ""

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON object from text that might contain other content.
        
        Args:
            text: Text possibly containing JSON
            
        Returns:
            str: Extracted JSON string
        """
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
        return text

    def _empty_fir_template(self) -> Dict:
        """Returns an empty FIR JSON structure as fallback."""
        return {
            "FIR": {
                "District": "", "Year": "", "FIR_No": "", "Date": "", "Police_Station": "", "Section": "",
                "Occurrence_of_Offense": {"Day": "", "Date": "", "Time_From": ""},
                "Information_Received_At_PS": {"Date": "", "Time": ""},
                "General_Diary_Reference": {"Entry_No": "", "Date": "", "Time": ""},
                "Type_of_Information": "",
                "Place_of_Occurrence": {
                    "Direction_and_Distance_From_PS": "", "Beat_No": "", "Place": "",
                    "Street_Village": "", "Area_Mandal": "", "City_District": "", "State": ""
                }
            },
            "Complainant_Informant": {
                "Name": "", "Father_Husband_Name": "", "Age": "", "Nationality": "", "Occupation": "",
                "Address": {
                    "Flat_No": "", "Street_Village": "", "Area_Mandal": "", "City_District": "", "State": ""
                },
                "Phone_No": ""
            },
            "Accused": {
                "Serial_No": "", "Name": "", "Cab_No": "", "Nationality": "", "Management": ""
            },
            "Victim": {
                "Name": "", "Father_Name": "", "Age": "", "Occupation": "",
                "Address": {
                    "Flat_No": "", "Street_Village": "", "Area_Mandal": "", "City_District": ""
                },
                "Injuries": "", "Condition": "",
                "Treatment": {"Initial": "", "Shifted_To": ""}
            },
            "Incident_Details": {"Date": "", "Time": "", "Location": "", "Description": ""},
            "Investigation": {
                "Case_Registered_U_S": "", "Case_No": "",
                "Investigation_Officer": {"Rank": "", "Name": ""},
                "Officer_in_Charge": {"Name": "", "Rank": "", "Signature": ""},
                "Date_Dispatch_Court": ""
            },
            "Reasons_for_Delay": "",
            "Court": ""
        }

if __name__ == "__main__":
    # Example usage
    from ocr_extractor import OcrExtractor
    
    # Initialize the agent with your preferred model
    agent = OllamaParsingAgent(model="llama3")
    
    # Get OCR text from image
    ocr_text = OcrExtractor.ocr_paddle("output/png/sample_fir_page_1.png")
    
    # Parse the OCR text into structured FIR data
    fir_data = agent.parse_fir(ocr_text)
    
    # Print the result
    print(json.dumps(fir_data, indent=2))