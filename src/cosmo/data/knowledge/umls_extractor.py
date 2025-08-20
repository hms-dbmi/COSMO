#!/usr/bin/env python3
"""
UMLS Knowledge Enricher

Enriches NCI-enriched knowledge with UMLS Metathesaurus synonyms and definitions.
Uses UMLS codes from OncoTree JSON to fetch comprehensive terminology data.

Author: Philip Chikontwe

IMPORTANT: Requires UMLS API key from UTS (https://uts.nlm.nih.gov/uts/)

Usage:
    # Enrich brain knowledge tree with UMLS data
    python umls_extractor.py --input brain_KT_nci.json --output brain_KT_umls.json --api-key YOUR_API_KEY

"""

import requests
import argparse
import json
import os
import time
import re
import langdetect
from typing import Dict, List, Optional
from tqdm import tqdm

def clean_text(text: str) -> str:
    """Remove links, citations, and clean text"""
    if not text:
        return text
    
    # Remove HTTP/HTTPS links
    text = re.sub(r'https?://[^\s]+', '', text)
    
    # Remove citation references like [pmid:27179225], [doi:10.1234/example], etc.
    text = re.sub(r'\[(?:pmid|doi|isbn|url):[^\]]+\]', '', text, flags=re.IGNORECASE)
    
    # Remove other common citation patterns like [1], [Author, 2023], etc.
    text = re.sub(r'\[\d+\]', '', text)  # Remove [1], [23], etc.
    text = re.sub(r'\[[^\]]{1,50}\]', '', text)  # Remove short bracketed content
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_english_text(text: str) -> bool:
    """Check if text is in English"""
    try:
        return langdetect.detect(text) == 'en'
    except:
        return False

def is_similar_text(text1: str, text2: str, similarity_threshold: float = 0.5) -> bool:
    """Check if two texts are similar (first 128 chars or 50% overlap)"""
    if not text1 or not text2:
        return False
    
    # Clean texts first
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    # Use first 128 characters for comparison
    compare_len = min(128, len(text1), len(text2))
    if compare_len == 0:
        return False
    
    text1_sample = text1[:compare_len].lower()
    text2_sample = text2[:compare_len].lower()
    
    # Calculate character overlap
    common_chars = sum(1 for c1, c2 in zip(text1_sample, text2_sample) if c1 == c2)
    similarity = common_chars / compare_len
    
    return similarity >= similarity_threshold

def remove_similar_duplicates(texts: List[str]) -> List[str]:
    """Remove similar duplicates from list of texts"""
    if not texts:
        return texts
    
    unique_texts = []
    for text in texts:
        cleaned_text = clean_text(text)
        if not cleaned_text:  # Skip empty texts
            continue
            
        # Check if similar text already exists
        is_duplicate = False
        for existing_text in unique_texts:
            if is_similar_text(cleaned_text, existing_text):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_texts.append(cleaned_text)
    
    return unique_texts

def process_text(text: str, is_abbrev: bool = False) -> str:
    """Process text to proper format"""
    # First clean the text
    text = clean_text(text)
    if not text:
        return text
        
    # Keep original case only for abbreviations (all caps or short terms)
    if is_abbrev:
        if text.isupper() and len(text) <= 5:
            return text
    
    return text.lower()

class UMLSClient:
    """UMLS REST API Client"""
    
    BASE_URI = "https://uts-ws.nlm.nih.gov/rest"
    VERSION = "current"
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("UMLS API key is required. Get one at https://uts.nlm.nih.gov/uts/")
        self.api_key = api_key
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to UMLS API"""
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        url = f"{self.BASE_URI}{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return {}
    
    def get_concept(self, cui: str) -> Dict:
        """Get basic concept information"""
        endpoint = f"/content/{self.VERSION}/CUI/{cui}"
        return self._make_request(endpoint)
    
    def get_atoms(self, cui: str, page: int = 1) -> Dict:
        """Get atoms (terms) for a concept"""
        endpoint = f"/content/{self.VERSION}/CUI/{cui}/atoms"
        params = {
            'pageNumber': page,
            'pageSize': 100,
            'language': 'ENG'
        }
        return self._make_request(endpoint, params)
    
    def get_definitions(self, cui: str) -> Dict:
        """Get definitions for a concept"""
        endpoint = f"/content/{self.VERSION}/CUI/{cui}/definitions"
        return self._make_request(endpoint)
    
    def get_synonyms(self, cui: str) -> List[str]:
        """Get all synonyms for a concept"""
        synonyms = set()
        page = 1
        
        while True:
            response = self.get_atoms(cui, page)
            if not response or 'result' not in response:
                break
                
            result = response.get('result', [])
            if not result:
                break
            
            for atom in result:
                if atom.get('language') != 'ENG':
                    continue
                    
                name = atom.get('name', '').strip()
                if name:
                    # Check if it's an abbreviation (all caps or short)
                    is_abbrev = name.isupper() or len(name) <= 5
                    processed_name = process_text(name, is_abbrev)
                    if processed_name:
                        synonyms.add(processed_name)
            
            if not response.get('pageCount') or page >= response['pageCount']:
                break
            page += 1
            time.sleep(0.1)
        
        # Remove similar duplicates instead of just sorting
        return remove_similar_duplicates(list(synonyms))

def enrich_knowledge_tree_with_umls(knowledge_tree: Dict, umls_client: UMLSClient) -> Dict:
    """Enrich knowledge tree with UMLS data"""
    enriched_tree = knowledge_tree.copy()
    
    pbar = tqdm(enriched_tree.items(), desc="Enriching with UMLS data")
    
    for key, entry in pbar:
        umls_code = entry.get('umls')
        if umls_code:
            try:
                pbar.set_description(f"Processing UMLS {key}")
                
                # Add synonyms with similarity-based deduplication
                synonyms = umls_client.get_synonyms(umls_code)
                if synonyms:
                    # Convert existing synonyms to lowercase too
                    existing_syns = [process_text(syn, syn.isupper() or len(syn) <= 5) 
                                   for syn in entry.get('synonyms', [])]
                    all_syns = existing_syns + synonyms
                    entry['synonyms'] = remove_similar_duplicates(all_syns)
                
                # Add definitions with similarity-based deduplication
                defs_response = umls_client.get_definitions(umls_code)
                if defs_response and 'result' in defs_response:
                    english_defs = []
                    for defn in defs_response['result']:
                        if defn.get('value'):
                            text = defn['value'].strip()
                            if is_english_text(text):
                                # Clean text to remove citations and links
                                cleaned_text = clean_text(text.lower())
                                if cleaned_text:
                                    english_defs.append(cleaned_text)
                    
                    # Convert existing definitions to lowercase and clean them too
                    existing_defs = [clean_text(defn.lower()) for defn in entry.get('definitions', [])
                                   if clean_text(defn.lower())]
                    all_defs = existing_defs + english_defs
                    entry['definitions'] = remove_similar_duplicates(all_defs)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nError processing UMLS code {umls_code} for {key}: {str(e)}")
                continue
    
    return enriched_tree

class UMLSExtractor:
    """Main UMLS enrichment class"""
    
    def __init__(self, api_key: str):
        self.umls_client = UMLSClient(api_key=api_key)
    
    def enrich_knowledge_file(self, input_path: str, output_path: str = None) -> Dict:
        """Enrich knowledge tree from file and optionally save result"""
        
        # Load existing knowledge tree
        with open(input_path, 'r') as f:
            knowledge_tree = json.load(f)
        
        print(f"Loaded knowledge tree with {len(knowledge_tree)} entries")
        
        # Count entries with UMLS codes
        umls_entries = sum(1 for entry in knowledge_tree.values() if entry.get('umls'))
        print(f"Found {umls_entries} entries with UMLS codes to enrich")
        
        if umls_entries == 0:
            print("No UMLS codes found in knowledge tree. Nothing to enrich.")
            return knowledge_tree
        
        # Enrich with UMLS data
        enriched_tree = enrich_knowledge_tree_with_umls(knowledge_tree, self.umls_client)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(enriched_tree, f, indent=2)
            print(f"Saved enriched knowledge tree to: {output_path}")
        
        # Report enrichment statistics
        total_synonyms = sum(len(entry.get('synonyms', [])) for entry in enriched_tree.values())
        total_definitions = sum(len(entry.get('definitions', [])) for entry in enriched_tree.values())
        
        print(f"UMLS enrichment complete:")
        print(f"  - Total synonyms: {total_synonyms}")
        print(f"  - Total definitions: {total_definitions}")
        
        return enriched_tree

def main():
    parser = argparse.ArgumentParser(description='Enrich Knowledge Tree with UMLS data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input knowledge tree JSON file (from NCI enrichment)')
    parser.add_argument('--output', type=str,
                        help='Output enriched JSON file (default: input_umls.json)')
    parser.add_argument('--api-key', type=str, required=True,
                        help='UMLS API key from UTS (https://uts.nlm.nih.gov/uts/)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        # Replace _nci.json with _umls.json or add _umls.json
        if args.input.endswith('_nci.json'):
            args.output = args.input.replace('_nci.json', '_umls.json')
        else:
            base_name = os.path.splitext(args.input)[0]
            args.output = f"{base_name}_umls.json"
    
    try:
        # Initialize UMLS enricher
        enricher = UMLSExtractor(api_key=args.api_key)
        
        # Enrich knowledge tree
        enriched_tree = enricher.enrich_knowledge_file(args.input, args.output)
        
        print(f"\nSuccessfully enriched knowledge tree with UMLS data")
        print(f"IMPORTANT: Keep your UMLS API key secure and do not share it!")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())