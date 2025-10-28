"""
Create a seed dataset for training.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from training import CADDataset


def main():
    """Create seed dataset."""
    print("Creating seed dataset...")

    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Create seed prompts
    output_path = os.path.join(data_dir, 'seed_prompts.json')
    CADDataset.create_seed_dataset(output_path)

    print(f"\nSeed dataset created at: {output_path}")
    print("\nYou can now:")
    print("  1. Expand the dataset with more prompts")
    print("  2. Add reference implementations")
    print("  3. Use it for training with train.py")


if __name__ == "__main__":
    main()
