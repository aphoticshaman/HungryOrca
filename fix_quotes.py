#!/usr/bin/env python3
"""Fix smart quotes in JavaScript files"""
import os
import sys

def fix_smart_quotes(filepath):
    """Replace smart quotes with regular quotes"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace smart quotes
        original = content
        content = content.replace("'", "'")  # left single quote
        content = content.replace("'", "'")  # right single quote
        content = content.replace(""", '"')  # left double quote
        content = content.replace(""", '"')  # right double quote
        content = content.replace("‚", "'")  # low single quote
        content = content.replace("„", '"')  # low double quote

        # Only write if changed
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    src_dir = '/home/user/HungryOrca/quantum_tarot/mobile/quantum-tarot-mvp/src'
    fixed_count = 0

    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.endswith('.js'):
                filepath = os.path.join(root, filename)
                if fix_smart_quotes(filepath):
                    fixed_count += 1

    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
