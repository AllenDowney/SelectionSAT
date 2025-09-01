#!/usr/bin/env python3
"""
Improved script to parse SAT Mathematics Percentile Ranks PDF into a pandas DataFrame.
"""

import pdfplumber
import pandas as pd
import re
from pathlib import Path

def parse_text(pdf_path):
    """
    Parse the SAT Mathematics Percentile Ranks PDF using text extraction.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        pandas.DataFrame: DataFrame with all extracted data
    """
    all_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Processing PDF with {len(pdf.pages)} pages...")
        
        for page_num, page in enumerate(pdf.pages):
            print(f"Processing page {page_num + 1}...")
            text = page.extract_text()
            all_text += text + "\n"
            print(f"  Page {page_num + 1}: {len(text)} characters")
    
    print(f"Total text length: {len(all_text)} characters")
    
    # Parse the text to extract score data
    data_rows = parse_text_to_rows(all_text)
    
    if data_rows:
        df = pd.DataFrame(data_rows)
        print(f"Extracted {len(df)} rows from text")
        return df
    else:
        print("No data rows extracted from text")
        return None

def parse_text_to_rows(text):
    """
    Parse text to extract score data rows.
    
    Args:
        text (str): Raw text from PDF
        
    Returns:
        list: List of data dictionaries
    """
    # Split text into lines
    lines = text.split('\n')
    print(f"Total lines in text: {len(lines)}")
    
    data_rows = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Split line into parts
        parts = line.split()
        if len(parts) >= 7:  # Need at least 7 parts: score, total_num, total_pct, male_num, male_pct, female_num, female_pct
            first_part = parts[0]
            
            # Check if first part is a 3-digit number (score)
            if first_part.isdigit() and len(first_part) == 3 and 200 <= int(first_part) <= 800:
                try:
                    score = int(first_part)
                    total_num = int(parts[1].replace(',', ''))
                    male_num = int(parts[3].replace(',', ''))
                    female_num = int(parts[5].replace(',', ''))
                    
                    data_rows.append({
                        'Score': score,
                        'Total_Number': total_num,
                        'Male_Number': male_num,
                        'Female_Number': female_num
                    })
                    
                    print(f"Line {i}: Found score {score} with data {total_num}, {male_num}, {female_num}")
                except (ValueError, IndexError) as e:
                    print(f"Line {i}: Error parsing '{line}': {e}")
                    continue
    
    print(f"Found {len(data_rows)} valid score rows")
    
    # Sort by score (descending)
    data_rows.sort(key=lambda x: x['Score'], reverse=True)
    
    return data_rows

def clean_score(score_str):
    """Clean and convert score to integer."""
    if score_str is None:
        return None
    
    try:
        # Remove any non-numeric characters and convert to int
        score = re.sub(r'[^\d]', '', str(score_str))
        return int(score) if score else None
    except:
        return None

def clean_number(num_str):
    """Clean and convert number to integer."""
    if num_str is None:
        return None
    
    try:
        # Remove commas and any non-numeric characters
        num = re.sub(r'[^\d]', '', str(num_str))
        return int(num) if num else None
    except:
        return None

def clean_percentile(pct_str):
    """Clean and convert percentile to integer."""
    if pct_str is None:
        return None
    
    try:
        # Extract just the percentile number
        pct = re.sub(r'[^\d]', '', str(pct_str))
        return int(pct) if pct else None
    except:
        return None

def parse_sat_percentiles(pdf_path, year):
    """
    Parse the SAT Mathematics Percentile Ranks PDF using text extraction.
    
    Args:
        pdf_path (str): Path to the PDF file
        year (int): Year of the data for naming output files
        
    Returns:
        pandas.DataFrame: DataFrame with all extracted data
    """
    # Parse the PDF using text extraction
    df = parse_text(pdf_path)
    
    if df is None or df.empty:
        print(f"No data was extracted from the {year} PDF.")
        return None
    
    print(f"\nSuccessfully extracted data from {year}!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    # Show data types
    print(f"\nData types:")
    print(df.dtypes)
    
    # Show statistics
    print(f"\nScore range: {df['Score'].min()} - {df['Score'].max()}")
    print(f"Total students: {df['Total_Number'].sum():,}")
    print(f"Male students: {df['Male_Number'].sum():,}")
    print(f"Female students: {df['Female_Number'].sum():,}")
    
    # Save to CSV with year in filename
    csv_filename = f"sat_math_{year}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved to: {csv_filename}")
    
    return df
            

if __name__ == "__main__":
    files = {
        2011: "SAT-Mathemathics_Percentile_Ranks_2011.pdf",
        2012: "SAT-Mathemathics-Percentile-Ranks-2012.pdf",
        2013: "SAT-Mathematics-Percentile-Ranks-2013.pdf",
        2014: "sat-percentile-ranks-mathematics-2014.pdf",
        2015: "sat-percentile-ranks-mathematics-2015.pdf",
    }

    for year, pdf_path in files.items():
        # Check if file exists before processing
        if not Path(pdf_path).exists():
            print(f"Warning: {pdf_path} not found, skipping...")
            continue
            
        print(f"\n{'='*80}")
        print(f"PROCESSING {year} DATA")
        print(f"{'='*80}")
        
        result = parse_sat_percentiles(pdf_path, year)
        if result is not None:
            print(f"Successfully processed {year} data!")
        else:
            print(f"Failed to process {year} data.")
        