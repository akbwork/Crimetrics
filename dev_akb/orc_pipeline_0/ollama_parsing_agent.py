import json
import requests
from typing import Dict, List, Union
from time_it import time_it

class OllamaParsingAgent:
    """Uses Ollama to parse and structure OCR text from FIR documents."""

    def __init__(self, model="deepseek-r1:latest", api_base="http://localhost:11434"):
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

        # Define the prompt for Ollama with specific emphasis on capturing full description
        prompt = f"""
You are an expert legal document parser. Extract the following FIR fields from the OCR text below. 
If a field is not found, use an empty string or "NA". Return a valid JSON object matching the structure and field names below. 
Do not include explanations, only the JSON.

IMPORTANT INSTRUCTIONS:
1. For the "Incident_Details.Description" field, extract the FULL detailed description of the incident.
2. DO NOT summarize or truncate the description - include ALL details exactly as written.
3. The description is often a paragraph or multiple sentences describing what happened.
4. Look for text that starts with dates/times and describes the sequence of events.

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
    "Description": ""  // EXTRACT FULL PARAGRAPH, DO NOT SUMMARIZE
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

Return ONLY the JSON object above, filled with extracted values. For the incident description, include the complete text without any summarization or truncation.
"""

        # Call Ollama API
        response = self._call_ollama(prompt)
        
        # Parse and validate the response
        json_str = self._extract_json(response)
        
        # Attempt to parse the JSON
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to fix common issues
            json_str = self._fix_json_issues(json_str)
            result = json.loads(json_str)
        
        # Post-process the result to ensure the description is fully captured
        result = self._post_process_description(result, text)
        
        return result

    def _call_ollama(self, prompt: str) -> str:
        """
        Call the Ollama API with the given prompt.
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            str: The response text from Ollama
        """
        url = f"{self.api_base}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Add parameters to encourage longer outputs
            "temperature": 0.1,  # Lower temperature for more deterministic output
            "num_predict": 4096  # Request more tokens for longer descriptions
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")

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
    
    def _fix_json_issues(self, json_str: str) -> str:
        """
        Fix common JSON formatting issues that might cause parsing to fail.
        
        Args:
            json_str: The JSON string to fix
            
        Returns:
            str: Fixed JSON string
        """
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix unescaped quotes in strings
        in_string = False
        result = []
        i = 0
        while i < len(json_str):
            char = json_str[i]
            if char == '"' and (i == 0 or json_str[i-1] != '\\'):
                in_string = not in_string
            
            if in_string and char == '"' and i > 0 and json_str[i-1] != '\\' and i < len(json_str)-1 and json_str[i+1] != ':' and json_str[i+1] != ',' and json_str[i+1] != '}' and json_str[i+1] != ']':
                result.append('\\"')
            else:
                result.append(char)
            
            i += 1
            
        return ''.join(result)
    
    def _post_process_description(self, result: Dict, original_text: str) -> Dict:
        """
        Attempt to improve incident description extraction by searching for it directly
        in the original text if it seems truncated or missing.
        
        Args:
            result: The parsed FIR data
            original_text: The original OCR text
            
        Returns:
            Dict: Updated FIR data with better description if found
        """
        import re
        
        # Check if the description is missing or suspiciously short
        incident_details = result.get("Incident_Details", {})
        description = incident_details.get("Description", "")
        
        if not description or len(description) < 50:
            # Try to find the description in the original text
            # Pattern 1: Look for text starting with a date and time pattern
            date_time_patterns = [
                r"On\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s+at\s+about\s+\d{1,2}:\d{2}\s+hrs,\s+(.*?)(?=\n\n|\Z)",
                r"On\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s+at\s+about\s+\d{1,2}\s+[AP]M,\s+(.*?)(?=\n\n|\Z)",
                r"On\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s+at\s+about\s+\d{1,2}\s+hrs,\s+(.*?)(?=\n\n|\Z)"
            ]
            
            for pattern in date_time_patterns:
                matches = re.search(pattern, original_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    incident_details["Description"] = matches.group(1).strip()
                    break
                    
            # Pattern 2: Look for sections labeled as description/facts/statement
            if not incident_details.get("Description"):
                label_patterns = [
                    r"(?:brief\s+facts|incident\s+details|description\s+of\s+offence|statement\s+of\s+complainant)[:\s]+(.*?)(?=\n\n|\Z)",
                    r"(?:facts\s+of\s+the\s+case|gist\s+of\s+the\s+case)[:\s]+(.*?)(?=\n\n|\Z)"
                ]
                
                for pattern in label_patterns:
                    matches = re.search(pattern, original_text, re.IGNORECASE | re.DOTALL)
                    if matches:
                        incident_details["Description"] = matches.group(1).strip()
                        break
            
            # Pattern 3: If we found a date and place but no description, look for text following them
            if not incident_details.get("Description") and incident_details.get("Date") and incident_details.get("Location"):
                date_str = incident_details.get("Date")
                location_str = incident_details.get("Location")
                if date_str and location_str:
                    context_pattern = f"(?:{re.escape(date_str)}|{re.escape(location_str)}).*?([^.]{100,})"
                    matches = re.search(context_pattern, original_text, re.IGNORECASE | re.DOTALL)
                    if matches:
                        incident_details["Description"] = matches.group(1).strip()
            
            # Update the result
            result["Incident_Details"] = incident_details
            
        return result

    def _empty_fir_template(self) -> Dict:
        """Returns an empty FIR JSON structure as fallback."""
        return {
            "FIR": {
                "District": "", "Year": "", "FIR_No": "", "Date": "", "Police_Station": "", "Section": "",
                "Occurrence_of_Offense": {"Day": "", "Date": "", "Time_From": ""},
                "Information_Received_At_PS": {"Date": "", "Time": ""},
                "General_Diary_Reference": {"Entry_No": "", "Date": "", "Time": ""},
                "Type_of_Information": "",
                "Place_of_Occurrence": {"Direction_and_Distance_From_PS": "", "Beat_No": "", "Place": "",
                                      "Street_Village": "", "Area_Mandal": "", "City_District": "", "State": ""}
            },
            "Complainant_Informant": {
                "Name": "", "Father_Husband_Name": "", "Age": "", "Nationality": "", "Occupation": "",
                "Address": {"Flat_No": "", "Street_Village": "", "Area_Mandal": "", "City_District": "", "State": ""},
                "Phone_No": ""
            },
            "Accused": {"Serial_No": "", "Name": "", "Cab_No": "", "Nationality": "", "Management": ""},
            "Victim": {
                "Name": "", "Father_Name": "", "Age": "", "Occupation": "",
                "Address": {"Flat_No": "", "Street_Village": "", "Area_Mandal": "", "City_District": ""},
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
    import re  # Add this import for regex support
    
    # Initialize the agent with your preferred model
    agent = OllamaParsingAgent(model="llama3")
    
    # Get OCR text from image
    ocr_text = OcrExtractor.ocr_paddle("output/png/sample_fir_page_1.png")
    
    # Parse the OCR text into structured FIR data
    fir_data = agent.parse_fir(ocr_text)
    
    # Print the result
    print(json.dumps(fir_data, indent=2))