#!/usr/bin/env python3
"""
Audit lab files to find ones with placeholder or minimal content.
"""

import os
import re
from pathlib import Path

# Placeholder patterns to search for
PLACEHOLDER_PATTERNS = [
    r'\[Detailed setup instructions\]',
    r'\[Step-by-step implementation\]',
    r'\[Verification steps\]',
    r'\[Expected output\]',
    r'\[Challenge description\]',
    r'\[Solution\]',
    r'TODO:',
    r'PLACEHOLDER',
    r'Coming soon',
    r'To be added',
]

# Minimum content threshold (lines)
MIN_CONTENT_LINES = 50

def check_lab_file(file_path):
    """Check if a lab file has placeholder content or is too short."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        # Check for placeholder patterns
        placeholders_found = []
        for pattern in PLACEHOLDER_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                placeholders_found.append(pattern)
        
        # Check content length
        non_empty_lines = [line for line in lines if line.strip()]
        is_too_short = len(non_empty_lines) < MIN_CONTENT_LINES
        
        return {
            'has_placeholders': len(placeholders_found) > 0,
            'placeholders': placeholders_found,
            'is_too_short': is_too_short,
            'line_count': len(non_empty_lines)
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    base_path = Path(r'H:\My Drive\Codes & Repos\codes_repos\devops-course')
    
    incomplete_labs = []
    total_labs = 0
    
    # Search all phases
    for phase_dir in ['phase1_beginner', 'phase2_intermediate', 'phase3_advanced']:
        phase_path = base_path / phase_dir
        if not phase_path.exists():
            continue
            
        # Search all module directories
        for module_dir in sorted(phase_path.glob('module-*')):
            labs_dir = module_dir / 'labs'
            if not labs_dir.exists():
                continue
                
            # Check all lab files
            for lab_file in sorted(labs_dir.glob('lab-*.md')):
                total_labs += 1
                result = check_lab_file(lab_file)
                
                if 'error' in result:
                    print(f"Error reading {lab_file}: {result['error']}")
                    continue
                
                if result['has_placeholders'] or result['is_too_short']:
                    incomplete_labs.append({
                        'file': str(lab_file),
                        'module': module_dir.name,
                        'phase': phase_dir,
                        'result': result
                    })
    
    # Print results
    print(f"\n{'='*80}")
    print(f"LAB FILE AUDIT RESULTS")
    print(f"{'='*80}\n")
    print(f"Total lab files scanned: {total_labs}")
    print(f"Incomplete lab files found: {len(incomplete_labs)}\n")
    
    if incomplete_labs:
        print(f"{'='*80}")
        print("INCOMPLETE LAB FILES:")
        print(f"{'='*80}\n")
        
        current_phase = None
        for lab in incomplete_labs:
            if current_phase != lab['phase']:
                current_phase = lab['phase']
                print(f"\n--- {current_phase.upper()} ---\n")
            
            print(f"Module: {lab['module']}")
            print(f"File: {Path(lab['file']).name}")
            
            if lab['result']['has_placeholders']:
                print(f"  ⚠️  Contains placeholders: {len(lab['result']['placeholders'])}")
            if lab['result']['is_too_short']:
                print(f"  ⚠️  Too short: {lab['result']['line_count']} lines (min: {MIN_CONTENT_LINES})")
            print()
    else:
        print("✅ All lab files have complete content!")
    
    # Save to file
    output_file = base_path / 'incomplete_labs.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Incomplete Lab Files Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total labs: {total_labs}\n")
        f.write(f"Incomplete: {len(incomplete_labs)}\n\n")
        
        for lab in incomplete_labs:
            f.write(f"{lab['file']}\n")
            if lab['result']['has_placeholders']:
                f.write(f"  - Placeholders: {lab['result']['placeholders']}\n")
            if lab['result']['is_too_short']:
                f.write(f"  - Lines: {lab['result']['line_count']}\n")
            f.write("\n")
    
    print(f"\nReport saved to: {output_file}")

if __name__ == '__main__':
    main()
