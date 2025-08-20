#!/usr/bin/env python3
"""
Knowledge CSV Generator

Converts enriched knowledge JSON to training CSV files following the original COSMO format.
Creates multiple CSV files for different knowledge types (synonyms, definitions, histology, cytology).

Author: Philip Chikontwe

Usage:
    # Generate all CSV types for brain
    python csv_generator.py --input brain_KT_umls.json --tissue brain --output-dir examples/data/

    # Generate specific CSV types only
    python csv_generator.py --input lung_KT_path.json --tissue lung --csv-types train syn def

"""

import json
import csv
import argparse
import os
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class CSVGenerator:
    """Generate training CSV files from enriched knowledge JSON"""
    
    # Tissue code mapping for CSV filenames
    TISSUE_CODES = {
        'brain': 'brainNIH',
        'lung': 'lungNIH', 
        'kidney': 'kidneyNIH',
        'CNS/Brain': 'brainNIH',
        'Lung': 'lungNIH',
        'Kidney': 'kidneyNIH'
    }
    
    # Tissue type codes for instance naming
    TISSUE_TYPE_CODES = {
        'brain': 't008',
        'lung': 't014', 
        'kidney': 't012',
        'CNS/Brain': 't008',
        'Lung': 't014',
        'Kidney': 't012'
    }
    
    def __init__(self, random_seed: int = 42):
        """Initialize with random seed for reproducible sampling"""
        random.seed(random_seed)
    
    def get_tissue_code(self, tissue_name: str) -> str:
        """Get tissue code for filename"""
        return self.TISSUE_CODES.get(tissue_name.lower(), tissue_name.lower() + 'NIH')
    
    def get_tissue_type_code(self, tissue_name: str) -> str:
        """Get tissue type code for instance naming"""
        return self.TISSUE_TYPE_CODES.get(tissue_name.lower(), 't999')
    
    def has_meaningful_content(self, entry: Dict) -> bool:
        """Check if entry has meaningful content (codes or text data)"""
        # Must have either NCI or UMLS code
        has_codes = bool(entry.get('nci', '').strip() or entry.get('umls', '').strip())
        
        # Or must have some text content
        has_synonyms = bool(entry.get('synonyms', []))
        has_definitions = bool(entry.get('definitions', []))
        has_histology = bool(entry.get('histologic_features', '').strip())
        has_cytology = bool(entry.get('cytologic_features', '').strip())
        has_text_content = has_synonyms or has_definitions or has_histology or has_cytology
        
        return has_codes #or has_text_content
    
    def sample_text_for_missing_features(self, definitions: List[str], num_samples: int = 1) -> List[str]:
        """Sample from definitions when histological/cytological features are missing"""
        if not definitions:
            return []
        
        # If we have fewer definitions than requested samples, return all
        if len(definitions) <= num_samples:
            return definitions
        
        # Randomly sample the requested number
        return random.sample(definitions, num_samples)
    
    def create_instance_name(self, entry_idx: int, tissue_type_code: str, suffix: str) -> str:
        """Create instance name following the original format: {entry_idx}_{tissue_type_code}_{suffix}"""
        return f"{entry_idx}_{tissue_type_code}_{suffix}"
    
    def generate_train_csv(self, knowledge_tree: Dict, tissue_code: str, tissue_type_code: str, output_dir: str) -> str:
        """Generate main training CSV with all available content (synonyms, definitions, histology, cytology)"""
        csv_data = []
        entry_idx = 0
        
        for key, entry in knowledge_tree.items():
            # Skip if this is just a tissue entry
            if key == entry.get('tissue', '').lower():
                continue
            
            # Skip entries with no meaningful content
            if not self.has_meaningful_content(entry):
                continue
            
            # Main term
            main_name = self.create_instance_name(entry_idx, tissue_type_code, 'main')
            csv_data.append([main_name, key])
            
            # Add synonyms
            synonyms = entry.get('synonyms', [])
            for syn_idx, synonym in enumerate(synonyms):
                if synonym.strip():
                    syn_name = self.create_instance_name(entry_idx, tissue_type_code, f'syn{syn_idx}')
                    csv_data.append([syn_name, synonym.strip()])
            
            # Add definitions
            definitions = entry.get('definitions', [])
            for def_idx, definition in enumerate(definitions):
                if definition.strip():
                    def_name = self.create_instance_name(entry_idx, tissue_type_code, f'def{def_idx}')
                    csv_data.append([def_name, definition.strip()])
            
            # Add histological features if available
            histologic_features = entry.get('histologic_features', '').strip()
            if histologic_features:
                hist_name = self.create_instance_name(entry_idx, tissue_type_code, 'histology')
                csv_data.append([hist_name, histologic_features])
            else:
                # Fallback: sample from definitions
                definitions  = entry.get('definitions', [])
                sampled_defs = self.sample_text_for_missing_features(definitions, 1)
                if sampled_defs:
                    hist_name = self.create_instance_name(entry_idx, tissue_type_code, 'histology')
                    csv_data.append([hist_name, sampled_defs[0]])
            
            # Add cytological features if available
            cytologic_features = entry.get('cytologic_features', '').strip()
            if cytologic_features:
                cyt_name = self.create_instance_name(entry_idx, tissue_type_code, 'cytology')
                csv_data.append([cyt_name, cytologic_features])
            else:
                # Fallback: sample from definitions
                definitions = entry.get('definitions', [])
                sampled_defs = self.sample_text_for_missing_features(definitions, 1)
                if sampled_defs:
                    cyt_name = self.create_instance_name(entry_idx, tissue_type_code, 'cytology')
                    csv_data.append([cyt_name, sampled_defs[0]])
            
            entry_idx += 1
        
        # Write CSV
        output_path = os.path.join(output_dir, f"{tissue_code}_knowledge_train.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['instance_name', 'text'])
            writer.writerows(csv_data)
        
        return output_path
    
    def generate_synonyms_csv(self, knowledge_tree: Dict, tissue_code: str, tissue_type_code: str, output_dir: str) -> str:
        """Generate synonyms test CSV"""
        csv_data = []
        entry_idx = 0
        
        for key, entry in knowledge_tree.items():
            if key == entry.get('tissue', '').lower():
                continue
            
            # Skip entries with no meaningful content
            if not self.has_meaningful_content(entry):
                continue
            
            # Main term
            main_name = self.create_instance_name(entry_idx, tissue_type_code, 'main')
            csv_data.append([main_name, key])
            
            # Add synonyms (same as train for now)
            synonyms = entry.get('synonyms', [])
            for syn_idx, synonym in enumerate(synonyms):
                if synonym.strip():
                    syn_name = self.create_instance_name(entry_idx, tissue_type_code, f'syn{syn_idx}')
                    csv_data.append([syn_name, synonym.strip()])
            
            entry_idx += 1
        
        output_path = os.path.join(output_dir, f"{tissue_code}_knowledge_syn_test.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['instance_name', 'text'])
            writer.writerows(csv_data)
        
        return output_path
    
    def generate_definitions_csv(self, knowledge_tree: Dict, tissue_code: str, tissue_type_code: str, output_dir: str) -> str:
        """Generate definitions test CSV"""
        csv_data = []
        entry_idx = 0
        
        for key, entry in knowledge_tree.items():
            if key == entry.get('tissue', '').lower():
                continue
            
            # Skip entries with no meaningful content
            if not self.has_meaningful_content(entry):
                continue
            
            # Main term
            main_name = self.create_instance_name(entry_idx, tissue_type_code, 'main')
            csv_data.append([main_name, key])
            
            # Add definitions
            definitions = entry.get('definitions', [])
            for def_idx, definition in enumerate(definitions):
                if definition.strip():
                    def_name = self.create_instance_name(entry_idx, tissue_type_code, f'def{def_idx}')
                    csv_data.append([def_name, definition.strip()])
            
            entry_idx += 1
        
        output_path = os.path.join(output_dir, f"{tissue_code}_knowledge_def_test.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['instance_name', 'text'])
            writer.writerows(csv_data)
        
        return output_path
    
    def generate_histology_csv(self, knowledge_tree: Dict, tissue_code: str, tissue_type_code: str, output_dir: str) -> str:
        """Generate histology test CSV"""
        csv_data = []
        entry_idx = 0
        
        for key, entry in knowledge_tree.items():
            if key == entry.get('tissue', '').lower():
                continue
            
            # Skip entries with no meaningful content
            if not self.has_meaningful_content(entry):
                continue
            
            # Main term
            main_name = self.create_instance_name(entry_idx, tissue_type_code, 'main')
            csv_data.append([main_name, key])
            
            # Add histological features if available
            histologic_features = entry.get('histologic_features', '')
            if histologic_features.strip():
                hist_name = self.create_instance_name(entry_idx, tissue_type_code, 'histology')
                csv_data.append([hist_name, histologic_features.strip()])
            else:
                # Fallback: sample from definitions
                definitions  = entry.get('definitions', [])
                sampled_defs = self.sample_text_for_missing_features(definitions, 1)
                if sampled_defs:
                    hist_name = self.create_instance_name(entry_idx, tissue_type_code, 'histology')
                    csv_data.append([hist_name, sampled_defs[0]])
            
            entry_idx += 1
        
        output_path = os.path.join(output_dir, f"{tissue_code}_knowledge_his_test.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['instance_name', 'text'])
            writer.writerows(csv_data)
        
        return output_path
    
    def generate_cytology_csv(self, knowledge_tree: Dict, tissue_code: str, tissue_type_code: str, output_dir: str) -> str:
        """Generate cytology test CSV"""
        csv_data = []
        entry_idx = 0
        
        for key, entry in knowledge_tree.items():
            if key == entry.get('tissue', '').lower():
                continue
            
            # Skip entries with no meaningful content
            if not self.has_meaningful_content(entry):
                continue
            
            # Main term
            main_name = self.create_instance_name(entry_idx, tissue_type_code, 'main')
            csv_data.append([main_name, key])
            
            # Add cytological features if available
            cytologic_features = entry.get('cytologic_features', '')
            if cytologic_features.strip():
                cyt_name = self.create_instance_name(entry_idx, tissue_type_code, 'cytology')
                csv_data.append([cyt_name, cytologic_features.strip()])
            else:
                # Fallback: sample from definitions
                definitions = entry.get('definitions', [])
                sampled_defs = self.sample_text_for_missing_features(definitions, 1)
                if sampled_defs:
                    cyt_name = self.create_instance_name(entry_idx, tissue_type_code, 'cytology')
                    csv_data.append([cyt_name, sampled_defs[0]])
            
            entry_idx += 1
        
        output_path = os.path.join(output_dir, f"{tissue_code}_knowledge_cyt_test.csv")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['instance_name', 'text'])
            writer.writerows(csv_data)
        
        return output_path
    
    def generate_all_csvs(self, input_path: str, tissue_name: str, output_dir: str, 
                         csv_types: List[str] = None) -> Dict[str, str]:
        """Generate all requested CSV files"""
        
        # Load knowledge tree
        with open(input_path, 'r') as f:
            knowledge_tree = json.load(f)
        
        print(f"Loaded knowledge tree with {len(knowledge_tree)} entries")
        
        # Count meaningful entries (filter out empty ones)
        meaningful_entries = 0
        empty_entries = []
        for key, entry in knowledge_tree.items():
            # Skip tissue entry
            if key == entry.get('tissue', '').lower():
                continue
            if self.has_meaningful_content(entry):
                meaningful_entries += 1
            else:
                empty_entries.append(key)
        
        print(f"Found {meaningful_entries} entries with meaningful content")
        if empty_entries:
            print(f"Filtered out {len(empty_entries)} empty entries: {empty_entries[:5]}{'...' if len(empty_entries) > 5 else ''}")
        
        # Get tissue code for filenames and instance names
        tissue_code = self.get_tissue_code(tissue_name)
        tissue_type_code = self.get_tissue_type_code(tissue_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Default to all CSV types if none specified
        if csv_types is None:
            csv_types = ['train', 'syn', 'def', 'his', 'cyt']
        
        # Generate CSVs
        generated_files = {}
        
        if 'train' in csv_types:
            path = self.generate_train_csv(knowledge_tree, tissue_code, tissue_type_code, output_dir)
            generated_files['train'] = path
            print(f"Generated training CSV: {path}")
        
        if 'syn' in csv_types:
            path = self.generate_synonyms_csv(knowledge_tree, tissue_code, tissue_type_code, output_dir)
            generated_files['synonyms'] = path
            print(f"Generated synonyms CSV: {path}")
        
        if 'def' in csv_types:
            path = self.generate_definitions_csv(knowledge_tree, tissue_code, tissue_type_code, output_dir)
            generated_files['definitions'] = path
            print(f"Generated definitions CSV: {path}")
        
        if 'his' in csv_types:
            path = self.generate_histology_csv(knowledge_tree, tissue_code, tissue_type_code, output_dir)
            generated_files['histology'] = path
            print(f"Generated histology CSV: {path}")
        
        if 'cyt' in csv_types:
            path = self.generate_cytology_csv(knowledge_tree, tissue_code, tissue_type_code, output_dir)
            generated_files['cytology'] = path
            print(f"Generated cytology CSV: {path}")
        
        return generated_files


def main():
    parser = argparse.ArgumentParser(description='Generate training CSV files from enriched knowledge JSON')
    parser.add_argument('--input', type=str, required=True,
                        help='Input enriched knowledge JSON file')
    parser.add_argument('--tissue', type=str, required=True,
                        choices=['brain', 'lung', 'kidney'],
                        help='Tissue type for filename generation')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for CSV files')
    parser.add_argument('--csv-types', nargs='+', 
                        choices=['train', 'syn', 'def', 'his', 'cyt'],
                        help='CSV types to generate (default: all)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = CSVGenerator(random_seed=args.random_seed)
        
        # Generate CSV files
        generated_files = generator.generate_all_csvs(
            input_path=args.input,
            tissue_name=args.tissue,
            output_dir=args.output_dir,
            csv_types=args.csv_types
        )
        
        print(f"\nSuccessfully generated {len(generated_files)} CSV files:")
        for csv_type, path in generated_files.items():
            print(f"  {csv_type}: {path}")
        
        print(f"\nCSV files follow the original format:")
        print(f"- instance_name: {{entry_idx}}_{generator.get_tissue_type_code(args.tissue)}_{{suffix}}")
        print(f"- Missing histology/cytology features filled from definitions")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())