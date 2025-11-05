#!/usr/bin/env python3
"""
ARC Testing Interface Server
=============================

Runs the official ARC testing interface from fchollet/ARC-AGI.

This provides a visual web interface to:
- View ARC tasks
- Manually solve them
- Test your understanding of task patterns

Usage:
    python3 run_arc_interface.py

Then open: http://localhost:8000
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000
DIRECTORY = "arc_testing_interface"

class ARCHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    # Check if directory exists
    if not os.path.exists(DIRECTORY):
        print(f"❌ Directory '{DIRECTORY}' not found!")
        print("Run: cp -r ARC-AGI/apps arc_testing_interface")
        return

    # Check for ARC Prize 2025 data files
    data_files = {
        'training': 'arc-agi_training_challenges.json',
        'evaluation': 'arc-agi_evaluation_challenges.json',
        'test': 'arc-agi_test_challenges.json'
    }

    found_files = []
    missing_files = []

    for name, filename in data_files.items():
        if os.path.exists(filename):
            found_files.append(f"  ✅ {name}: {filename}")
        else:
            missing_files.append(f"  ❌ {name}: {filename}")

    print("="*70)
    print("🧩 ARC PRIZE 2025 - VISUAL TESTING INTERFACE")
    print("="*70)

    print("\n📁 ARC Dataset Status:")
    for f in found_files:
        print(f)
    if missing_files:
        print("\n⚠️  Missing files:")
        for f in missing_files:
            print(f)

    if not found_files:
        print("\n❌ No ARC data files found!")
        print("Please ensure you have arc-agi_*.json files in the current directory.")
        return

    print(f"\n🚀 Starting server on http://localhost:{PORT}")
    print("\n📖 How to use:")
    print("   1. Select a dataset (Training/Evaluation/Test)")
    print("   2. Browse and select a task from the list")
    print("   3. Study the training examples (input → output)")
    print("   4. Try to solve the test case yourself")
    print("   5. Submit your solution to check")
    print("\n💡 Tips:")
    print("   - Use 'Edit' mode to draw individual pixels")
    print("   - Use 'Select' mode to copy/paste rectangular regions")
    print("   - Use 'Flood fill' to fill connected areas with the same color")
    print("   - Click 'Copy from input' to start with the input grid")
    print("   - Resize the grid if the output should be a different size")
    print("\n🎨 Color Palette:")
    print("   0=Black, 1=Blue, 2=Red, 3=Green, 4=Yellow")
    print("   5=Gray, 6=Magenta, 7=Orange, 8=Sky, 9=Brown")
    print("\n⌨️  Press Ctrl+C to stop the server")
    print("="*70)

    # Start server
    with socketserver.TCPServer(("", PORT), ARCHandler) as httpd:
        url = f"http://localhost:{PORT}/local_data_loader.html"

        print(f"\n✅ Server running!")
        print(f"🌐 Open: {url}")
        print()

        # Try to open browser automatically
        try:
            webbrowser.open(url)
            print("🔗 Opening browser...")
        except:
            print("💻 Manually open the URL above in your browser")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
