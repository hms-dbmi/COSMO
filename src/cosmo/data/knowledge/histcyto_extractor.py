#!/usr/bin/env python3
"""
Histocytological Features Extractor

Enriches UMLS-enriched knowledge with histological and cytological features 
scraped from PathologyOutlines.com. Adds detailed pathology descriptions.

Author: Philip Chikontwe

Usage:
    # Enrich brain knowledge tree with pathology features
    python histcyto_extractor.py --input brain_KT_umls.json --output brain_KT_path.json
    
    # Debug mode with specific tumors only
    python histcyto_extractor.py --input brain_KT_umls.json --output brain_KT_path.json --debug

"""

import requests
from bs4 import BeautifulSoup
import json
import argparse
import time
from typing import Dict, List, Optional, Tuple
import re
import os
from urllib.parse import quote, urljoin
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class PathologyOutlinesScraper:
    """Scraper for PathologyOutlines.com"""
    
    BASE_URL = "https://www.pathologyoutlines.com"
    
    def __init__(self, delay: float = 1.0, max_retries: int = 3, timeout: int = 30):
        self.delay = delay
        self.cache = {}
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(['GET', 'HEAD'])
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_maxsize=10,
            pool_block=False
        )
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.timeout = timeout

    def get_tumor_features(self, tumor_name: str, tissue_type: str = "") -> Dict[str, str]:
        """Get features for a specific tumor"""
        try:
            url = self.construct_url(tumor_name, tissue_type)
            print(f"Fetching: {url}")

            # Check cache first
            if url in self.cache:
                return self.cache[url]

            try:
                # Use a tuple for connect and read timeouts
                response = self.session.get(url, timeout=(5, self.timeout))
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                features = {
                    "histologic_features": self.get_section_content(soup, "Microscopic (histologic) description"),
                    "cytologic_features": self.get_section_content(soup, "Cytology description")
                }
                
                self.cache[url] = features
                time.sleep(self.delay)
                return features
                
            except requests.exceptions.RequestException as e:
                print(f"\nError fetching {url}: {str(e)}")
                return {"histologic_features": "", "cytologic_features": ""}
                
        except Exception as e:
            print(f"\nError processing {tumor_name}: {str(e)}")
            return {"histologic_features": "", "cytologic_features": ""}
    
    def clean_text(self, text: str) -> str:
        """Clean and format text content"""
        if not text:
            return ""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove references
        text = re.sub(r'\([^)]*\d+[^)]*\)', '', text)
        # Remove other unwanted patterns
        text = re.sub(r'contributed by.*$', '', text.lower(), flags=re.IGNORECASE)
        return text.strip()

    def extract_list_content(self, ul_element) -> str:
        """Extract content from a list element"""
        if not ul_element:
            return ""
        
        content = []
        for li in ul_element.find_all('li', recursive=True):
            text = li.get_text(strip=True)
            if text and not any(skip in text.lower() for skip in ['reference:', 'contributed by']):
                content.append(text)
        
        return ' '.join(content)

    def get_section_content(self, soup: BeautifulSoup, section_title: str) -> str:
        """Extract content from a specific section"""
        content = []
        
        # Find the section div that contains the title
        for div in soup.find_all('div', class_='block_section'):
            title_div = div.find('div', class_='topicheading_title')
            if title_div and section_title.lower() in title_div.text.strip().lower():
                content_div = div.find('div', class_='block_body')
                if content_div:
                    # Extract content from unordered lists
                    for ul in content_div.find_all('ul', recursive=False):
                        list_content = self.extract_list_content(ul)
                        if list_content:
                            content.append(list_content)
        
        return self.clean_text(' '.join(content))

    def get_tissue_path(self, tissue_type: str) -> str:
        """Get the base path for a tissue type"""
        # Map tissue names to PathologyOutlines URL paths
        tissue_paths = {
            'Adrenal Gland': 'adrenal',
            'Ampulla of Vater': 'ampulla',
            'Biliary Tract': 'gallbladder',  # Maps to gallbladder & extrahepatic bile ducts section
            'Bladder/Urinary Tract': ['bladder','urinary'],
            'Bone': 'bone', 
            'Bowel': 'colon',  # Note: Also includes smallbowel but mainly maps to colon
            'CNS/Brain': 'cnstumor',
            'Breast': 'breast',
            'Cervix': 'cervix',
            'Eye': 'eye',
            'Head and Neck': 'oralcavity',  # Maps to oral cavity/oropharynx section
            'Kidney': 'kidney',
            'Liver': 'liver',
            'Lung': 'lungtumor',
            'Lymphoid': 'lymph',
            'Myeloid': 'marrow',
            'Other': '',  # Generic mapping
            'Ovary/Fallopian Tube': ['ovary','fallopian'],  # Note: Also includes fallopiantubes
            'Pancreas': 'pancreas',
            'Penis': 'penis',
            'Peritoneum': 'peritoneum',
            'Pleura': 'pleura',
            'Peripheral Nervous System': 'nerve',
            'Prostate': 'prostate',
            'Skin': 'skin',
            'Soft Tissue': 'softtissue',
            'Esophagus/Stomach': 'stomach',  # Note: Also includes esophagus
            'Testis': 'testis',
            'Thymus': 'thymus',
            'Thyroid': 'thyroid',
            'Uterus': 'uterus',
            'Vulva/Vagina': ['vulva','vagina']  # Note: Also includes vagina
        }
        
        # Clean up tissue type string
        tissue_type = tissue_type.strip()
        
        # Return the mapped path or empty string if not found
        return tissue_paths.get(tissue_type, '')

    def construct_url(self, tumor_name: str, tissue_type: str = "") -> str:
        """Construct URL based on tumor name and tissue type"""
        # Remove special characters and spaces
        tumor_slug = re.sub(r'[^\w\s-]', '', tumor_name.lower())
        tumor_slug = re.sub(r'[-\s]+', '', tumor_slug)
        
        # Get tissue path
        base_path = self.get_tissue_path(tissue_type)
        
        # Handle multiple possible paths
        if isinstance(base_path, list):
            # Try each possible path
            for path in base_path:
                url = f"{self.BASE_URL}/topic/{path}{tumor_slug}.html"
                try:
                    # Use a shorter timeout for HEAD requests
                    response = self.session.head(url, timeout=10)
                    if response.status_code == 200:
                        return url
                except:
                    continue
            
            # If no path works, default to first one
            base_path = base_path[0]
        
        # Use default 'topic' path if no specific tissue path found
        if not base_path:
            return f"{self.BASE_URL}/topic/{tumor_slug}.html"
        
        return f"{self.BASE_URL}/topic/{base_path}{tumor_slug}.html"


class HistcytoExtractor:
    """Main histocytological features extractor"""
    
    def __init__(self, delay: float = 1.0, max_retries: int = 3, timeout: int = 30):
        self.scraper = PathologyOutlinesScraper(delay=delay, max_retries=max_retries, timeout=timeout)
    
    def enrich_knowledge_file(self, input_path: str, output_path: str = None, debug: bool = False) -> Dict:
        """Enrich knowledge tree with pathology features from file"""
        
        try:
            # Load knowledge tree
            with open(input_path, 'r') as f:
                knowledge_tree = json.load(f)
            
            print(f"Loaded knowledge tree with {len(knowledge_tree)} entries")
            
            if debug:
                print("Debug mode enabled - processing limited entries...")
                # Test with specific tumors for debugging
                debug_tumors = ["ependymoma"]
                knowledge_tree = {k: v for k, v in knowledge_tree.items() if k in debug_tumors}
                print(f"Debug mode: processing {len(knowledge_tree)} entries")
            
            # Create output directory if needed
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Process each entry
            pbar = tqdm(knowledge_tree.items(), desc="Processing pathology features")
            processed_count = 0
            features_added = 0
            
            for key, entry in pbar:
                # Skip main tissue types (they are usually too general)
                if 'tissue' in entry and entry.get('tissue', '').lower() == key.lower():
                    continue
                
                pbar.set_description(f"Processing {key}")
                
                # Get tissue type from entry
                tissue_type = entry.get('tissue', '')
                
                # Get features from PathologyOutlines
                features = self.scraper.get_tumor_features(key, tissue_type)
                
                # Add features if found
                if features['histologic_features']:
                    entry['histologic_features'] = features['histologic_features']
                    features_added += 1
                if features['cytologic_features']:
                    entry['cytologic_features'] = features['cytologic_features']
                    features_added += 1
                
                processed_count += 1
            
            print(f"\nPathology enrichment complete:")
            print(f"  - Entries processed: {processed_count}")
            print(f"  - Features added: {features_added}")
            
            # Save enriched tree
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(knowledge_tree, f, indent=2)
                print(f"Saved enriched knowledge tree to: {output_path}")
            
            return knowledge_tree
            
        except Exception as e:
            print(f"\nError processing knowledge tree: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Scrape pathology features from PathologyOutlines.com')
    parser.add_argument('--input', type=str, required=True,
                        help='Input knowledge tree JSON file (from UMLS enrichment)')
    parser.add_argument('--output', type=str,
                        help='Output enriched JSON file (default: input_path.json)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (process limited entries)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Request timeout in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        # Replace _umls.json with _path.json or add _path.json
        if args.input.endswith('_umls.json'):
            args.output = args.input.replace('_umls.json', '_path.json')
        else:
            base_name   = os.path.splitext(args.input)[0]
            args.output = f"{base_name}_path.json"
    
    try:
        # Initialize extractor with custom settings
        extractor = HistcytoExtractor(
            delay=args.delay,
            max_retries=3,
            timeout=args.timeout
        )
        
        # Enrich knowledge tree
        enriched_tree = extractor.enrich_knowledge_file(
            args.input, 
            args.output, 
            args.debug
        )
        
        print(f"\nSuccessfully enriched knowledge tree with pathology features")
        print(f"IMPORTANT: Please be respectful of PathologyOutlines.com servers")
        print(f"Consider using --delay option to increase delays between requests")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())