#!/usr/bin/env python3
"""
Debug script to investigate configuration duplication issue.
"""

import json
import os
import sys

from extract_captions import load_provider_config, CaptionExtractor

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_config():
    """Debug configuration loading and status generation for troubleshooting."""
    print("=== CONFIGURATION DEBUG SESSION ===")
    
    # Test provider configuration loading
    print("\n1. Testing provider configuration loading for 'gemini':")
    config = load_provider_config('gemini')
    print("Configuration keys:", list(config.keys()))
    print("Client configuration:", config.get('client_config', {}))
    
    # Add required keys for testing
    config['db_path'] = 'captions.db'
    config['root_dir'] = 'test_dir'
    
    # Test CaptionExtractor initialization
    print("\n2. Testing CaptionExtractor initialization:")
    try:
        extractor = CaptionExtractor(config)
        print("Extractor configuration keys:", list(extractor.config.keys()))
        print("Extractor client configuration:", extractor.config.get('client_config', {}))
        
        # Test status retrieval
        print("\n3. Testing status retrieval:")
        status = extractor.get_status()
        print("Status keys:", list(status.keys()))
        
        # Analyze configuration in status for potential duplicates
        status_config = status.get('config', {})
        print("Status configuration keys:", list(status_config.keys()))
        print("Status client configuration:", status_config.get('client_config', {}))
        
        # Detailed analysis for debugging
        print("\n4. Detailed type analysis:")
        print("Status object type:", type(status))
        print("Status configuration type:", type(status_config))
        
        # Analyze client configuration for duplicate entries
        client_config = status_config.get('client_config', {})
        print("Client configuration type:", type(client_config))
        print("Client configuration model:", client_config.get('model', 'NOT_FOUND'))
        
        # Display raw string representation for debugging
        print("\n5. Raw string representation:")
        print(repr(status_config))
        
        # Generate formatted JSON output
        print("\n6. JSON output:")
        json_output = json.dumps(status, indent=2, ensure_ascii=False)
        print(json_output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config()