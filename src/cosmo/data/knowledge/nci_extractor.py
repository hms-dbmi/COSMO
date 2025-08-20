#!/usr/bin/env python3
"""
NCI Thesaurus Knowledge Enricher

Enriches OncoTree knowledge with NCI Thesaurus synonyms and definitions using reference codes.
Uses the existing NCI codes from OncoTree JSON to fetch detailed information.

Author: Philip Chikontwe

Usage:
    # Enrich brain knowledge tree with NCI data
    python nci_extractor.py --input brain_KT.json --output brain_KT_nci.json
    
    # With optional API key (not required but may help with rate limits)
    python nci_extractor.py --input brain_KT.json --output brain_KT_nci.json --api-key YOUR_KEY

"""

import requests
import argparse
import json
import os
import time
import re
from typing import Dict, List, Optional
from tqdm import tqdm

class NCIThesaurus:
    """NCI Thesaurus API client"""
    
    BASE_URL = "https://api-evsrest.nci.nih.gov/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {
            'accept': 'application/json',
            'apiKey': api_key if api_key else ''
        }
    
    def clean_text(self, text: str) -> str:
        """Remove links and clean text"""
        if not text:
            return text
        # Remove HTTP/HTTPS links
        text = re.sub(r'https?://[^\s]+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def is_similar_text(self, text1: str, text2: str, similarity_threshold: float = 0.5) -> bool:
        """Check if two texts are similar (first 128 chars or 50% overlap)"""
        if not text1 or not text2:
            return False
        
        # Clean texts first
        text1 = self.clean_text(text1)
        text2 = self.clean_text(text2)
        
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
    
    def remove_similar_duplicates(self, texts: List[str]) -> List[str]:
        """Remove similar duplicates from list of texts"""
        if not texts:
            return texts
        
        unique_texts = []
        for text in texts:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:  # Skip empty texts
                continue
                
            # Check if similar text already exists
            is_duplicate = False
            for existing_text in unique_texts:
                if self.is_similar_text(cleaned_text, existing_text):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_texts.append(cleaned_text)
        
        return unique_texts
    
    def get_concept_by_code(self, nci_code: str) -> Dict:
        """Fetch concept details by NCI code"""
        url = f"{self.BASE_URL}/concept/ncit/{nci_code}?include=summary"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return {}
    
    def get_synonyms(self, nci_code: str) -> List[str]:
        """Get all synonyms for a concept"""
        concept = self.get_concept_by_code(nci_code)
        synonyms = []
        if concept and 'synonyms' in concept:
            for syn in concept['synonyms']:
                if 'name' in syn:
                    # Convert to lowercase and add to list
                    synonyms.append(syn['name'].lower())
                    # If it's an abbreviation (shorter than 5 chars), add the original case too
                    if len(syn['name']) < 5:
                        synonyms.append(syn['name'])
        
        # Remove similar duplicates instead of just exact duplicates
        return synonyms #self.remove_similar_duplicates(synonyms)
    
    def get_definitions(self, nci_code: str) -> List[str]:
        """Get all definitions for a concept"""
        concept = self.get_concept_by_code(nci_code)
        definitions = []
        if concept and 'definitions' in concept:
            for defn in concept['definitions']:
                if 'definition' in defn:
                    # Convert definition to lowercase
                    definitions.append(defn['definition'].lower())
        
        # Remove similar duplicates instead of just exact duplicates
        return self.remove_similar_duplicates(definitions)


def enrich_knowledge_tree_with_nci(knowledge_tree: Dict, nci_client: NCIThesaurus) -> Dict:
    """Enrich knowledge tree with NCI Thesaurus data"""
    enriched_tree = knowledge_tree.copy()
    
    # Create progress bar
    pbar = tqdm(enriched_tree.items(), desc="Enriching entries")
    
    for key, entry in pbar:
        nci_code = entry.get('nci')
        if nci_code:
            try:
                pbar.set_description(f"Processing {key}")
                
                # Add synonyms with similarity-based deduplication
                synonyms = nci_client.get_synonyms(nci_code)
                if synonyms:
                    # Combine existing and new synonyms, then remove similar duplicates
                    all_synonyms = entry['synonyms'] + synonyms
                    entry['synonyms'] = nci_client.remove_similar_duplicates(all_synonyms)
                
                # Add definitions with similarity-based deduplication  
                definitions = nci_client.get_definitions(nci_code)
                if definitions:
                    # Combine existing and new definitions, then remove similar duplicates
                    all_definitions = entry['definitions'] + definitions
                    entry['definitions'] = nci_client.remove_similar_duplicates(all_definitions)
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nError processing NCI code {nci_code} for {key}: {str(e)}")
    
    return enriched_tree


class NCIExtractor:
    """Main NCI enrichment class"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.nci_client = NCIThesaurus(api_key=api_key)
    
    def enrich_knowledge_file(self, input_path: str, output_path: str = None) -> Dict:
        """Enrich knowledge tree from file and optionally save result"""
        
        # Load existing knowledge tree
        with open(input_path, 'r') as f:
            knowledge_tree = json.load(f)
        
        print(f"Loaded knowledge tree with {len(knowledge_tree)} entries")
        
        # Count entries with NCI codes
        nci_entries = sum(1 for entry in knowledge_tree.values() if entry.get('nci'))
        print(f"Found {nci_entries} entries with NCI codes to enrich")
        
        if nci_entries == 0:
            print("No NCI codes found in knowledge tree. Nothing to enrich.")
            return knowledge_tree
        
        # Enrich with NCI data
        enriched_tree = enrich_knowledge_tree_with_nci(knowledge_tree, self.nci_client)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(enriched_tree, f, indent=2)
            print(f"Saved enriched knowledge tree to: {output_path}")
        
        # Report enrichment statistics
        total_synonyms = sum(len(entry.get('synonyms', [])) for entry in enriched_tree.values())
        total_definitions = sum(len(entry.get('definitions', [])) for entry in enriched_tree.values())
        
        print(f"Enrichment complete:")
        print(f"  - Total synonyms added: {total_synonyms}")
        print(f"  - Total definitions added: {total_definitions}")
        
        return enriched_tree


def main():
    parser = argparse.ArgumentParser(description='Enrich Knowledge Tree with NCI Thesaurus data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input knowledge tree JSON file (from OncoTree)')
    parser.add_argument('--output', type=str,
                        help='Output enriched JSON file (default: input_nci.json)')
    parser.add_argument('--api-key', type=str, 
                        help='NCI EVS API key (optional, may help with rate limits)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_nci.json"
    
    try:
        # Initialize NCI enricher
        enricher = NCIExtractor(api_key=args.api_key)
        
        # Enrich knowledge tree
        enriched_tree = enricher.enrich_knowledge_file(args.input, args.output)
        
        print(f"\nSuccessfully enriched knowledge tree with NCI data")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())