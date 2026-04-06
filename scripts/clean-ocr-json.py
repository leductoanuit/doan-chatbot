#!/usr/bin/env python3
"""
Clean OCR JSON: Remove unnecessary characters and fix common OCR errors
"""
import json
import re
from pathlib import Path

# Common OCR artifacts and unnecessary characters to remove/fix
OCR_FIXES = {
    # Common OCR misreads
    r'\bÔ\b': 'ô',
    r'\bÓ\b': 'ó',
    r'\bOO\b': 'ô',
    r'\boo\b': 'ô',
    r'Oo\b': '',
    r'\bOo\b': '',
    # Remove repeated special characters
    r'_+': ' ',
    r'-{2,}': '-',
    # Clean up strange diacritics
    r'ď': 'd',
    r'ť': 't',
    r'ň': 'n',
    r'ì': 'i',
    r'í': 'í',
    # Remove noise patterns
    r'\s{2,}': ' ',
    r'\n{2,}': '\n',
}

def clean_text(text):
    """Clean OCR text by removing artifacts and fixing common errors"""
    if not text:
        return text

    # Apply regex fixes
    for pattern, replacement in OCR_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    # Remove common OCR noise patterns
    # Lines that are just symbols
    text = re.sub(r'^\s*[_\-=\*\.]+\s*$', '', text, flags=re.MULTILINE)

    # Fix common letter substitutions
    replacements = {
        'đđ': 'đ',
        'óó': 'ó',
        'óo': 'ô',
        'à¤': 'ă',
        'ằ': 'ă',
        'àƒ': 'à',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove lines that are just random symbols or mostly non-Vietnamese characters
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Skip lines that are mostly symbols
        symbol_count = sum(1 for c in line if c in '_-=*@#$%^&()[]{}|<>,.:;')
        if symbol_count / len(line) > 0.5 and len(line) < 5:
            continue
        # Skip lines that look like OCR artifacts (only numbers/symbols)
        if re.match(r'^[\d\s\-\._,]*$', line):
            continue
        cleaned_lines.append(line.strip())

    text = '\n'.join(cleaned_lines)

    # Final cleanup
    text = text.strip()

    return text

def process_json_file(input_path, output_path):
    """Process JSON file and clean OCR content"""
    print(f"Reading {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} documents...")

    cleaned_data = []
    for i, doc in enumerate(data):
        if 'content' in doc:
            original_length = len(doc['content'])
            doc['content'] = clean_text(doc['content'])
            cleaned_length = len(doc['content'])
            print(f"  Doc {i+1}: {original_length} → {cleaned_length} chars")

        cleaned_data.append(doc)

    print(f"\nWriting cleaned data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print("✓ Done!")

if __name__ == '__main__':
    input_file = Path('/Users/cps/do an chatbot/data/processed/all_documents_ocr.json')
    output_file = Path('/Users/cps/do an chatbot/data/processed/all_documents_ocr_cleaned.json')

    process_json_file(input_file, output_file)
