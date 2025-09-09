import re
import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import PyPDF2
from datetime import datetime

class HSBCEarningsParser:
    """Final HSBC earnings parser with corrected role/company logic and answer numbering"""
    
    def __init__(self):
        # Known HSBC executives with their roles - used to identify internal speakers
        self.hsbc_executives = {
            'noel quinn': 'Group Chief Executive',
            'georges elhedery': 'Group Chief Executive',  
            'pam kaur': 'Group Chief Financial Officer',
            'ewen stevenson': 'Group Chief Financial Officer', 
            'richard holmes': 'Group Head of Investor Relations',
            'richard o\'connor': 'Global Head of Investor Relations',
            'stuart tait': 'Group Chief Risk and Compliance Officer',
            'barry o\'byrne': 'Chief Executive, Global Commercial Banking',
            'surendra rosha': 'Chief Executive, Wealth and Personal Banking',
            'greg guyett': 'Chief Executive, Global Banking and Markets',
            'jose carvalho': 'Chief Executive, Global Banking',
            'colin bell': 'Chief Executive, HSBC UK',
            'david liao': 'Chief Executive, HSBC Asia Pacific',
            'jon bingham': 'Interim Group Chief Financial Officer',
            'mark tucker': 'Group Chairman'
        }
        
        # Common analyst firm names for identification
        self.analyst_firms = [
            'Morgan Stanley', 'JP Morgan', 'JPMorgan', 'Goldman Sachs', 'Barclays', 
            'UBS', 'Credit Suisse', 'Deutsche Bank', 'Bank of America', 'Citigroup', 
            'Citi', 'BNP Paribas', 'Societe Generale', 'RBC', 'BMO', 'Nomura',
            'Jefferies', 'Berenberg', 'Autonomous', 'KBW', 'Numis', 'Mediobanca',
            'CICC', 'Redburn Atlantic', 'China Securities', 'RBC Capital Markets',
            'Autonomous Research', 'Deutsche Numis', 'BNP Paribas Exane'
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def parse_filename_metadata(self, filename: str) -> Dict[str, Any]:
        """Extract year, quarter information from filename"""
        # Handle HSBC naming conventions
        filename_lower = filename.lower()
        
        # After: Simple string matching
        if '2023' in filename_lower:
            year = 2023
        elif '2024' in filename_lower:
            year = 2024
        elif '2025' in filename_lower:
            year = 2025
        else:
            year = None
        
        # Map HSBC quarter conventions
        if any(q in filename_lower for q in ['1q', 'q1', 'quarter1', 'first']):
            quarter = 'Q1'
        elif any(q in filename_lower for q in ['2q', 'q2', 'interim', 'half', 'h1']):
            quarter = 'Q2'
        elif any(q in filename_lower for q in ['3q', 'q3', 'quarter3', 'third']):
            quarter = 'Q3'  
        elif any(q in filename_lower for q in ['4q', 'q4', 'annual', 'full', 'year']):
            quarter = 'Q4'
        else:
            quarter = 'Unknown'
            
        return {'year': year, 'quarter': quarter}

    def identify_section_boundaries(self, text: str) -> Tuple[int, Optional[int]]:
        """Find where presentation ends and Q&A begins"""
        presentation_start = 0
        qa_start = None
        
        # Look for Q&A section indicators
        qa_patterns = [
            r'(?:Questions?\s*(?:and|&)\s*Answers?|Q\s*&\s*A)',
            r'(?:Operator|Host).*(?:question|Q&A)',
            r'(?:We\s*will\s*now\s*(?:begin|start|take).*questions?)',
            r'(?:(?:Thank\s*you|Thanks).*(?:question|Q&A))'
        ]
        
        for pattern in qa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                qa_start = match.start()
                break
                
        # Alternative: look for first analyst question
        if qa_start is None:
            analyst_pattern = r'(?:Thank you|Thanks|Good morning|Hi)'
            match = re.search(analyst_pattern, text)
            if match:
                qa_start = match.start()
        
        return 0, qa_start

    def extract_speaker_info(self, speaker_line: str) -> Dict[str, str]:
        """Enhanced speaker extraction with corrected role/company logic"""
        # Clean the speaker line
        speaker_line = re.sub(r'^\s*[\-\*•]+\s*', '', speaker_line.strip())
        
        # Try the standard pattern: "NAME, TITLE/COMPANY:" 
        match = re.match(r'^([A-Z][A-Z\s\'\-\.]+?),\s*([^:]+?):\s*(.+)', speaker_line, re.IGNORECASE)
        
        if match:
            name = match.group(1).strip().title()
            title_company_part = match.group(2).strip()
            content_preview = match.group(3).strip()
            
            # Check if this is an HSBC executive by name
            name_key = name.lower()
            if name_key in self.hsbc_executives:
                return {
                    'speaker_name': name,
                    'role': self.hsbc_executives[name_key],  # Use specific job title as role
                    'company': 'HSBC',  # Company should always be HSBC for internal
                    'content_start': content_preview
                }
            
            # Check if it's an HSBC executive by title keywords in the title part
            if any(keyword in title_company_part.lower() for keyword in 
                   ['group', 'chief', 'ceo', 'cfo', 'head', 'interim', 'chairman']):
                return {
                    'speaker_name': name,
                    'role': title_company_part,  # Use the title as role
                    'company': 'HSBC',  # Company is HSBC for internal speakers
                    'content_start': content_preview
                }
            
            # Otherwise, it's an external analyst
            company = self.extract_company_name(title_company_part)
            return {
                'speaker_name': name,
                'role': 'Analyst',
                'company': company,
                'content_start': content_preview
            }
        
        # Try simple pattern for cases without comma: "NAME:"
        simple_match = re.match(r'^([A-Z][A-Za-z\s\'\-\.]+?):\s*(.+)', speaker_line, re.IGNORECASE)
        if simple_match:
            name = simple_match.group(1).strip().title()
            content_preview = simple_match.group(2).strip()
            
            # Check if known HSBC executive
            name_key = name.lower()
            if name_key in self.hsbc_executives:
                return {
                    'speaker_name': name,
                    'role': self.hsbc_executives[name_key],
                    'company': 'HSBC',
                    'content_start': content_preview
                }
        
        # Fallback - couldn't parse properly
        return {
            'speaker_name': '',
            'role': '',
            'company': '',
            'content_start': speaker_line
        }

    def extract_company_name(self, company_part: str) -> str:
        """Extract clean company name from company part of speaker line"""
        # Look for known analyst firms in the company part
        for firm in self.analyst_firms:
            if firm.lower() in company_part.lower():
                return firm
        
        # If no known firm found, clean up the company part
        # Remove common title words and return what's left
        company_clean = re.sub(r'\b(?:analyst|research|equity|managing|director|senior|vice|president)\b', 
                              '', company_part, flags=re.IGNORECASE)
        company_clean = re.sub(r'\s+', ' ', company_clean.strip())
        
        return company_clean if company_clean else company_part

    def simple_parse_transcript_content(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse transcript content with corrected logic"""
        records = []
        
        # Identify section boundaries  
        presentation_start, qa_start = self.identify_section_boundaries(text)
        
        if qa_start is None:
            sections = [('presentation', text)]
        else:
            sections = [
                ('presentation', text[presentation_start:qa_start]),
                ('qa', text[qa_start:])
            ]
        
        for section_type, section_text in sections:
            lines = section_text.split('\n')
            
            current_speaker_info = None
            current_content = []
            question_number = 0
            answer_number_for_current_question = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts a new speaker
                if re.match(r'^[A-Z][A-Za-z\s\',\-\.&]+:', line):
                    # Save previous speaker's content if any
                    if current_speaker_info and current_content:
                        self._save_speaker_record(
                            records, current_speaker_info, current_content, 
                            section_type, question_number, answer_number_for_current_question, metadata
                        )
                    
                    # Start new speaker
                    current_speaker_info = self.extract_speaker_info(line)
                    current_content = []
                    
                    # Update counters based on speaker type
                    if section_type == 'qa':
                        if current_speaker_info['role'] == 'Analyst':
                            # New question - increment question number and reset answer number
                            question_number += 1
                            answer_number_for_current_question = 0
                        elif current_speaker_info['company'] == 'HSBC':
                            # New answer to the current question
                            answer_number_for_current_question += 1
                    
                    # Add any content that was on the same line as speaker name
                    if current_speaker_info.get('content_start'):
                        current_content.append(current_speaker_info['content_start'])
                        
                else:
                    # Continue accumulating content for current speaker
                    if current_speaker_info:
                        current_content.append(line)
            
            # Don't forget the last speaker
            if current_speaker_info and current_content:
                self._save_speaker_record(
                    records, current_speaker_info, current_content,
                    section_type, question_number, answer_number_for_current_question, metadata
                )
        
        return records

    def _save_speaker_record(self, records: List, speaker_info: Dict, content: List, 
                           section_type: str, question_num: int, answer_num: int, metadata: Dict):
        """Helper to save a speaker's record with corrected numbering and role logic"""
        content_text = ' '.join(content).strip()
        
        # Skip very short content
        if len(content_text) < 10:
            return
        
        # Determine question/answer numbers based on section and speaker type
        if section_type == 'presentation':
            # No question/answer numbering in presentation
            final_question_num = None
            final_answer_num = None
        elif section_type == 'qa':
            if speaker_info['role'] == 'Analyst':
                # Analyst question
                final_question_num = question_num
                final_answer_num = None
            elif speaker_info['company'] == 'HSBC':
                # HSBC executive answer
                final_question_num = None
                final_answer_num = answer_num if answer_num > 0 else None
            else:
                final_question_num = None
                final_answer_num = None
        else:
            final_question_num = None
            final_answer_num = None
        
        # Fix role assignment based on section
        final_role = speaker_info['role']
        if speaker_info['company'] == 'HSBC':
            if section_type == 'presentation':
                # Keep specific job title for presentation section
                final_role = speaker_info['role']
            elif section_type == 'qa':
                # Use generic 'management' for Q&A section
                final_role = 'management'
            
        records.append({
            'section': section_type,
            'question_number': final_question_num,
            'answer_number': final_answer_num,
            'speaker_name': speaker_info['speaker_name'],
            'role': final_role,
            'company': speaker_info['company'],
            'content': content_text,
            'year': metadata['year'],
            'quarter': metadata['quarter'],
            'is_pleasantry': self.is_pleasantry(content_text),
            'source_pdf': metadata['filename']
        })

    def is_pleasantry(self, content: str) -> bool:
        """Determine if content is a pleasantry/greeting"""
        content_lower = content.lower().strip()
        
        pleasantry_patterns = [
            r'^(?:good (?:morning|afternoon|evening)|hi|hello|thanks?|thank you)(?:\s|$|\.)',
            r'^(?:thanks?|thank you)(?:\s|\.)',
            r'^(?:hi|hello)\s*(?:everyone|all)?(?:\s|$|\.)',
            r'^(?:good\s+(?:morning|afternoon)|thanks?\s*(?:very\s+much)?)\s*[,.]?\s*$',
            r'^(?:thank you for (?:taking|the))',
        ]
        
        for pattern in pleasantry_patterns:
            if re.match(pattern, content_lower):
                return True
                
        return len(content.split()) <= 8 and any(word in content_lower for word in ['thank', 'good', 'hi', 'hello'])

    def process_single_file(self, pdf_path: str) -> pd.DataFrame:
        """Process a single PDF file"""
        filename = os.path.basename(pdf_path)
        logging.info(f"Processing: {filename}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            logging.warning(f"No text extracted from {filename}")
            return pd.DataFrame()
        
        # Parse metadata
        metadata = self.parse_filename_metadata(filename)
        metadata['filename'] = filename
        
        # Parse content
        records = self.simple_parse_transcript_content(text, metadata)
        
        if not records:
            logging.warning(f"No records parsed from {filename}")
            return pd.DataFrame()
        
        return pd.DataFrame(records)

    def process_directory(self, input_dir: str, output_dir: str) -> pd.DataFrame:
        """Process all PDF files in directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        all_records = []
        
        # Process each PDF file
        for filename in sorted(os.listdir(input_dir)):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                df = self.process_single_file(pdf_path)
                
                if not df.empty:
                    all_records.append(df)
                    
                    # Save individual quarter file
                    quarter_filename = f"hsbc_{filename.replace('.pdf', '')}_parsed_final.csv"
                    quarter_path = os.path.join(output_dir, quarter_filename)
                    df.to_csv(quarter_path, index=False)
                    logging.info(f"Saved {len(df)} records to {quarter_filename}")
        
        # Combine all data
        if all_records:
            combined_df = pd.concat(all_records, ignore_index=True)
            
            # Save combined file
            combined_path = os.path.join(output_dir, 'all_hsbc_earnings_data.csv')
            combined_df.to_csv(combined_path, index=False)
            logging.info(f"Saved combined dataset: {len(combined_df)} records")
            
            return combined_df
        else:
            logging.warning("No data processed successfully")
            return pd.DataFrame()

# Example usage/test function
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    input_directory = "~/data/raw/hsbc"  # Update path as needed
    output_directory = "~/data/processed/hsbc"  # Update path as needed
    
    # Expand user paths
    input_directory = os.path.expanduser(input_directory)
    output_directory = os.path.expanduser(output_directory)
    
    if not os.path.exists(input_directory):
        logging.error(f"Input directory not found: {input_directory}")
        return
    
    try:
        parser = HSBCEarningsParser()
        df = parser.process_directory(input_directory, output_directory)
        
        if not df.empty:
            logging.info(f"Successfully processed {len(df)} total records")
            logging.info(f"Data shape: {df.shape}")
            logging.info(f"Years covered: {sorted(df['year'].unique())}")
            logging.info(f"Quarters covered: {sorted(df['quarter'].unique())}")
            
            # Show sample of data
            print("\nSample data:")
            print(df[['speaker_name', 'role', 'company', 'section']].head(10))
            
            # Show role distribution
            print("\nRole distribution:")
            print(df['role'].value_counts())
            
        else:
            logging.warning("No data was processed successfully")
            
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()