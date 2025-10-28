"""
Dataset management for CAD code generation training.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from torch.utils.data import Dataset
import random


@dataclass
class CADSample:
    """A single training sample."""
    prompt: str
    expected_dimensions: Optional[Dict[str, float]] = None
    reference_code: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CADDataset(Dataset):
    """
    Dataset for CAD code generation.

    Manages prompts and (optionally) reference implementations
    for training and evaluation.
    """

    def __init__(
        self,
        samples: List[CADSample],
        augment: bool = False
    ):
        """
        Initialize dataset.

        Args:
            samples: List of CAD samples
            augment: Whether to apply data augmentation
        """
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        prompt = sample.prompt

        # Apply augmentation if enabled
        if self.augment:
            prompt = self._augment_prompt(prompt)

        item = {
            'prompt': prompt,
            'expected_dimensions': sample.expected_dimensions,
        }

        if sample.reference_code:
            item['reference_code'] = sample.reference_code

        if sample.category:
            item['category'] = sample.category

        return item

    def _augment_prompt(self, prompt: str) -> str:
        """
        Apply data augmentation to prompt.

        Args:
            prompt: Original prompt

        Returns:
            Augmented prompt
        """
        # Simple augmentations:
        # 1. Add variations in phrasing
        # 2. Add detail specifications
        # 3. Reorder requirements

        prefixes = [
            "Create a 3D model of ",
            "Design ",
            "Build ",
            "Generate ",
            "Model ",
        ]

        # Randomly choose a prefix
        if random.random() < 0.5:
            prefix = random.choice(prefixes)
            # Remove common starting words
            for start in ["Create", "Design", "Build", "Generate", "Model", "A", "An"]:
                if prompt.startswith(start + " "):
                    prompt = prompt[len(start)+1:]
                    break
            prompt = prefix + prompt

        return prompt

    @classmethod
    def from_json(cls, json_path: str, augment: bool = False) -> 'CADDataset':
        """
        Load dataset from JSON file.

        Args:
            json_path: Path to JSON file
            augment: Whether to apply augmentation

        Returns:
            CADDataset instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = CADSample(
                prompt=item['prompt'],
                expected_dimensions=item.get('expected_dimensions'),
                reference_code=item.get('reference_code'),
                category=item.get('category'),
                metadata=item.get('metadata')
            )
            samples.append(sample)

        return cls(samples, augment=augment)

    @classmethod
    def create_seed_dataset(cls, output_path: str):
        """
        Create a seed dataset with common CAD objects.

        Args:
            output_path: Path to save JSON file
        """
        seed_data = [
            {
                "prompt": "Create a simple chair with four legs, a seat, and a backrest",
                "category": "furniture",
                "expected_dimensions": {
                    "width": 40.0,
                    "height": 85.0,
                    "depth": 40.0
                }
            },
            {
                "prompt": "Design a rectangular table with four cylindrical legs",
                "category": "furniture",
                "expected_dimensions": {
                    "width": 120.0,
                    "height": 75.0,
                    "depth": 80.0
                }
            },
            {
                "prompt": "Build a storage box with a lid",
                "category": "container",
                "expected_dimensions": {
                    "width": 30.0,
                    "height": 20.0,
                    "depth": 20.0
                }
            },
            {
                "prompt": "Create a cylindrical vase with a wider base",
                "category": "decorative",
                "expected_dimensions": {
                    "height": 25.0,
                    "diameter": 10.0
                }
            },
            {
                "prompt": "Design a bookshelf with three shelves",
                "category": "furniture",
                "expected_dimensions": {
                    "width": 80.0,
                    "height": 120.0,
                    "depth": 30.0
                }
            },
            {
                "prompt": "Build a simple desk with a drawer",
                "category": "furniture",
                "expected_dimensions": {
                    "width": 120.0,
                    "height": 75.0,
                    "depth": 60.0
                }
            },
            {
                "prompt": "Create a cylindrical pen holder",
                "category": "office",
                "expected_dimensions": {
                    "height": 10.0,
                    "diameter": 8.0
                }
            },
            {
                "prompt": "Design a simple stool with three legs",
                "category": "furniture",
                "expected_dimensions": {
                    "height": 45.0,
                    "diameter": 35.0
                }
            },
            {
                "prompt": "Build a rectangular planter box",
                "category": "garden",
                "expected_dimensions": {
                    "width": 60.0,
                    "height": 30.0,
                    "depth": 30.0
                }
            },
            {
                "prompt": "Create a simple cup with a handle",
                "category": "kitchenware",
                "expected_dimensions": {
                    "height": 10.0,
                    "diameter": 8.0
                }
            }
        ]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(seed_data, f, indent=2)

        print(f"Seed dataset created with {len(seed_data)} samples at {output_path}")


def split_dataset(
    dataset: CADDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple[CADDataset, CADDataset, CADDataset]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: CADDataset to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    samples = dataset.samples.copy()
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    return (
        CADDataset(train_samples, augment=dataset.augment),
        CADDataset(val_samples, augment=False),
        CADDataset(test_samples, augment=False)
    )
