#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Markdown format using jupytext.

This script finds all .ipynb files in the current directory and converts them
to Markdown format, saving the results in the md/ directory.
"""

import sys
from pathlib import Path

try:
    import jupytext
except ImportError:
    print("Error: jupytext is not installed.")
    print("Please install it with: pip install jupytext")
    sys.exit(1)


def convert_notebooks_to_md():
    """Convert all notebooks in the current directory to Markdown format."""
    # Get the script's directory (project root)
    project_root = Path(__file__).parent.resolve()
    
    # Define source and destination directories
    source_dir = project_root
    dest_dir = project_root / "md"
    
    # Create md directory if it doesn't exist
    dest_dir.mkdir(exist_ok=True)
    
    # Find all .ipynb files in the source directory
    notebooks = list(source_dir.glob("*.ipynb"))
    
    if not notebooks:
        print("No notebooks found in the current directory.")
        return
    
    print(f"Found {len(notebooks)} notebook(s) to convert:")
    
    # Convert each notebook
    for notebook_path in sorted(notebooks):
        notebook_name = notebook_path.stem
        output_path = dest_dir / f"{notebook_name}.md"
        
        try:
            print(f"  Converting {notebook_path.name} -> md/{output_path.name}")
            
            # Use jupytext to convert to markdown
            jupytext.write(
                jupytext.read(notebook_path),
                output_path,
                fmt="md"
            )
            
            print(f"    ✓ Successfully converted to {output_path}")
            
        except Exception as e:
            print(f"    ✗ Error converting {notebook_path.name}: {e}")
    
    print(f"\nConversion complete! Markdown files saved in {dest_dir}")


if __name__ == "__main__":
    convert_notebooks_to_md()

