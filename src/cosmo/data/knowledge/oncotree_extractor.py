#!/usr/bin/env python3
"""
OncoTree Knowledge Extractor

Extracts cancer type information from OncoTree API following the original COSMO design.
Creates structured JSON with UMLS and NCI reference codes for downstream processing.

Author: Philip Chikontwe

Usage:
    # List available tissues
    python oncotree_extractor.py --list-tissues
    
    # Generate knowledge tree for brain
    python oncotree_extractor.py --tissue BRAIN --output brain_KT.json

"""

import json
import sys
import urllib.request
import argparse
import os
import re
from typing import Dict, List, Optional

class OncoTreeExtractor:
    """OncoTree API client following original COSMO design"""
    
    API_URL_BASE = "http://oncotree.mskcc.org/api/"
    ENDPOINTS = {
        'tumor_types': 'tumorTypes',
        'main_types': 'mainTypes',
        'tree': 'tumorTypes/tree',
        'search': 'tumorTypes/search'
    }
    
    def __init__(self):
        pass
    
    def fetch_data(self, endpoint_url: str) -> Dict:
        """Generic function to fetch data from OncoTree API"""
        try:
            response = urllib.request.urlopen(endpoint_url)
            if response.getcode() != 200:
                sys.stderr.write(f"ERROR (HttpStatusCode {response.getcode()}): Unable to retrieve data.\n")
                sys.exit(1)
            return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            sys.stderr.write(f"ERROR: {str(e)}\n")
            sys.exit(1)
    
    def fetch_main_types(self) -> Dict:
        """Fetch main tumor types"""
        return self.fetch_data(self.API_URL_BASE + self.ENDPOINTS['main_types'])
    
    def fetch_tumor_types(self) -> Dict:
        """Fetch all tumor types"""
        return self.fetch_data(self.API_URL_BASE + self.ENDPOINTS['tumor_types'])
    
    def fetch_complete_tree(self) -> Dict:
        """Fetch complete tumor type tree"""
        return self.fetch_data(self.API_URL_BASE + self.ENDPOINTS['tree'])
    
    def get_available_tissues(self, tree: Dict) -> List[str]:
        """Get list of available tissue types from OncoTree"""
        if 'TISSUE' in tree and 'children' in tree['TISSUE']:
            return list(tree['TISSUE']['children'].keys())
        return []
    
    def clean_text(self, text: str) -> str:
        """Remove links and clean text"""
        if not text:
            return text
        # Remove HTTP/HTTPS links
        text = re.sub(r'https?://[^\s]+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def format_name_as_key(self, name: str) -> str:
        """Format a name into a proper key format"""
        return name.lower()
    
    def get_external_references(self, node: Dict) -> tuple:
        """Extract UMLS and NCI codes from node"""
        umls_code = ""
        nci_code = ""
        if 'externalReferences' in node:
            refs = node['externalReferences']
            if 'UMLS' in refs and refs['UMLS']:
                umls_code = refs['UMLS'][0]
            if 'NCI' in refs and refs['NCI']:
                nci_code = refs['NCI'][0]
        return umls_code, nci_code
    
    def process_tumor_node(self, node: Dict, parent_name: str = None, tissue_name: str = None) -> Dict:
        """Process a tumor node and its children recursively"""
        result = {}
        
        # Process current node
        node_name = node.get('name', '')
        node_key = self.format_name_as_key(node_name)
        umls_code, nci_code = self.get_external_references(node)
        
        # Build parents list
        parents = []
        if tissue_name and node_name != tissue_name:  # Add tissue as parent if not self-referential
            parents.append(tissue_name)
        if parent_name and parent_name != tissue_name:  # Add direct parent if different from tissue
            parents.append(parent_name)
        
        result[node_key] = {
            "umls": umls_code,
            "nci": nci_code,
            "tissue": node.get('tissue', ''),
            "parents": parents,
            "synonyms": [],
            "definitions": [],
            "histologic_features": "",
            "cytologic_features": ""
        }
        
        # Process children if they exist
        if 'children' in node and node['children']:
            for child_code, child_node in node['children'].items():
                child_data = self.process_tumor_node(child_node, node_name, tissue_name)
                result.update(child_data)
        
        return result
    
    def process_tissue_tumors(self, tissue_tree: Dict, tissue_code: str) -> Dict:
        """Process tumor types for a given tissue and create knowledge tree structure"""
        knowledge_tree = {}
        
        # Create the base tissue entry
        tissue_name = tissue_tree['name']
        tissue_umls, tissue_nci = self.get_external_references(tissue_tree)
        tissue_key = self.format_name_as_key(tissue_name)
        
        # Add tissue as root node
        knowledge_tree[tissue_key] = {
            "umls": tissue_umls,
            "nci": tissue_nci,
            "tissue": tissue_name,
            "synonyms": [],
            "definitions": [],
            "histologic_features": "",
            "cytologic_features": ""
        }
        
        # Process all tumors under this tissue
        if 'children' in tissue_tree:
            for tumor_code, tumor_node in tissue_tree['children'].items():
                tumor_data = self.process_tumor_node(tumor_node, tissue_name, tissue_name)
                knowledge_tree.update(tumor_data)
        
        return knowledge_tree
    
    def remove_duplicates(self, knowledge_tree: Dict) -> Dict:
        """Remove duplicate entries from knowledge tree"""
        seen_keys = set()
        cleaned_tree = {}
        
        for key, entry in knowledge_tree.items():
            # Create unique identifier based on key and UMLS/NCI codes
            identifier = (key, entry.get('umls', ''), entry.get('nci', ''))
            
            if identifier not in seen_keys:
                seen_keys.add(identifier)
                # Clean text fields
                cleaned_entry = entry.copy()
                if 'definitions' in cleaned_entry:
                    cleaned_entry['definitions'] = [self.clean_text(d) for d in cleaned_entry['definitions']]
                if 'synonyms' in cleaned_entry:
                    cleaned_entry['synonyms'] = [self.clean_text(s) for s in cleaned_entry['synonyms']]
                if 'histologic_features' in cleaned_entry:
                    cleaned_entry['histologic_features'] = self.clean_text(cleaned_entry['histologic_features'])
                if 'cytologic_features' in cleaned_entry:
                    cleaned_entry['cytologic_features'] = self.clean_text(cleaned_entry['cytologic_features'])
                
                cleaned_tree[key] = cleaned_entry
        
        return cleaned_tree
    
    def save_knowledge_tree(self, tree: Dict, filename: str):
        """Save knowledge tree to JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(tree, f, indent=2)
    
    def extract_tissue_knowledge(self, tissue_code: str, output_path: str = None) -> Dict:
        """Main method to extract knowledge for a tissue"""
        
        # Fetch OncoTree data
        print("Fetching OncoTree data...")
        main_types = self.fetch_main_types()
        tumor_types = self.fetch_tumor_types()
        complete_tree = self.fetch_complete_tree()
        
        # Validate tissue
        available_tissues = self.get_available_tissues(complete_tree)
        tissue_code = tissue_code.upper()
        
        if tissue_code not in available_tissues:
            available_str = ", ".join(sorted(available_tissues))
            raise ValueError(f"Invalid tissue type '{tissue_code}'. Available: {available_str}")
        
        # Process tumors for specified tissue
        print(f"Processing {tissue_code} tissue...")
        tissue_tree = complete_tree['TISSUE']['children'][tissue_code]
        knowledge_tree = self.process_tissue_tumors(tissue_tree, tissue_code)
        
        # Clean duplicates and links
        print("Removing duplicates and cleaning text...")
        knowledge_tree = self.remove_duplicates(knowledge_tree)
        
        # Save if output path provided
        if output_path:
            self.save_knowledge_tree(knowledge_tree, output_path)
            print(f"Saved knowledge tree to: {output_path}")
        
        print(f"Processed {len(knowledge_tree)} total entries for {tissue_code}")
        return knowledge_tree
    
    def list_available_tissues(self) -> Dict[str, str]:
        """List all available tissue types"""
        complete_tree = self.fetch_complete_tree()
        available_tissues = self.get_available_tissues(complete_tree)
        
        tissue_info = {}
        for tissue in sorted(available_tissues):
            tissue_name = complete_tree['TISSUE']['children'][tissue]['name']
            tissue_info[tissue] = tissue_name
            
        return tissue_info


def main():
    parser = argparse.ArgumentParser(description='Generate OncoTree Knowledge Tree for specific tissue')
    parser.add_argument('--tissue', type=str, help='Tissue code (e.g., BRAIN, BREAST, LUNG)')
    parser.add_argument('--list-tissues', action='store_true', help='List available tissue types')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    args = parser.parse_args()

    extractor = OncoTreeExtractor()
    
    # List available tissues if requested
    if args.list_tissues:
        print("\nAvailable tissue types:")
        tissues = extractor.list_available_tissues()
        for code, name in tissues.items():
            print(f"- {code}: {name}")
        return

    # Validate tissue argument
    if not args.tissue:
        print("Error: Please specify a tissue type with --tissue or use --list-tissues to see available options")
        return
    
    # Set default output path if not provided
    if not args.output:
        args.output = f"{args.tissue.lower()}_onco.json"
    
    try:
        # Extract knowledge
        knowledge_tree = extractor.extract_tissue_knowledge(args.tissue, args.output)
        print(f"\nGenerated knowledge tree for {args.tissue} tissue successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()