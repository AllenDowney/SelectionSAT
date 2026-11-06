#!/usr/bin/env python3
"""
Simple script to extract and print EVERY row from PDF tables with no filtering.
"""

import pdfplumber

def parse_raw(pdf_path):
    """Extract and print raw text from PDF without parsing."""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages")
        
        for page_num, page in enumerate(pdf.pages):
            print(f"\n{'='*80}")
            print(f"PAGE {page_num + 1}")
            print(f"{'='*80}")
            
            text = page.extract_text()
            print(f"Text length: {len(text)} characters")
            print("\nRAW TEXT (no parsing):")
            print(text)

if __name__ == "__main__":
    parse_raw("sat-percentile-ranks-mathematics-2014.pdf")
