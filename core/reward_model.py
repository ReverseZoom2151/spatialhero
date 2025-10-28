"""
Multi-modal reward model for evaluating generated CAD models.

This is the key improvement over the original architecture:
- Combines multiple evaluation signals
- Uses programmatic verification alongside visual evaluation
- Provides detailed, actionable feedback
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np
from openai import OpenAI

from core.verifier import CodeVerifier, ValidationResult
from core.renderer import CADRenderer
from utils.cad_utils import CADQueryExecutor


@dataclass
class RewardComponents:
    """Individual components of the reward signal."""
    code_valid: float  # 0.0 or 1.0
    execution_valid: float  # 0.0 or 1.0
    geometry_valid: float  # 0.0 or 1.0
    dimension_accuracy: float  # 0.0 to 1.0
    visual_quality: float  # 0.0 to 1.0
    topology_valid: float  # 0.0 or 1.0


@dataclass
class RewardResult:
    """Complete reward evaluation result."""
    total_reward: float  # Weighted combination
    components: RewardComponents
    feedback: str  # Natural language feedback
    validation: ValidationResult
    success: bool
    metadata: Optional[Dict[str, Any]] = None


class RewardModel:
    """
    Multi-modal reward model that combines:
    1. Code validity (syntax, execution)
    2. Geometric verification (topology, measurements)
    3. Visual quality (LLM-based evaluation)
    4. Dimensional accuracy (programmatic measurement)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        visual_model: str = "gpt-5-mini-2025-08-07",  # Changed from Pro due to API compatibility
        weights: Optional[Dict[str, float]] = None,
        use_visual_eval: bool = True
    ):
        """
        Initialize reward model.

        Args:
            api_key: OpenAI API key
            visual_model: Model to use for visual evaluation
            weights: Custom weights for reward components
            use_visual_eval: Whether to use LLM for visual evaluation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.visual_model = visual_model
        self.use_visual_eval = use_visual_eval

        # Default weights (should sum to 1.0)
        self.weights = weights or {
            'code_valid': 0.20,
            'dimension_accuracy': 0.30,
            'visual_quality': 0.30,
            'topology_valid': 0.20
        }

        # Validate weights
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights.values())}")

        self.verifier = CodeVerifier()
        self.renderer = CADRenderer()
        self.executor = CADQueryExecutor()

    def evaluate(
        self,
        code: str,
        prompt: str,
        expected_dimensions: Optional[Dict[str, float]] = None,
        render_views: Optional[List[str]] = None
    ) -> RewardResult:
        """
        Comprehensive evaluation of generated CAD code.

        Args:
            code: Generated CADQuery code
            prompt: Original natural language prompt
            expected_dimensions: Expected dimensions for verification
            render_views: Views to render for visual evaluation

        Returns:
            RewardResult with detailed evaluation
        """
        # 1. Verify code (syntax, execution, geometry)
        validation = self.verifier.verify(code, expected_dimensions)

        # Initialize components with validation results
        components = RewardComponents(
            code_valid=1.0 if validation.code_valid else 0.0,
            execution_valid=1.0 if validation.execution_valid else 0.0,
            geometry_valid=1.0 if validation.geometry_valid else 0.0,
            dimension_accuracy=0.0,
            visual_quality=0.0,
            topology_valid=1.0 if validation.geometry_valid else 0.0
        )

        feedback_parts = []

        # If code doesn't execute, return early with low reward
        if not validation.execution_valid:
            feedback = self._generate_failure_feedback(validation)
            total_reward = self._calculate_total_reward(components)
            return RewardResult(
                total_reward=total_reward,
                components=components,
                feedback=feedback,
                validation=validation,
                success=False
            )

        # 2. Evaluate dimensional accuracy
        if validation.measurements:
            if validation.measurements.get('success'):
                components.dimension_accuracy = validation.measurements.get('average_accuracy', 0.0)
                if components.dimension_accuracy < 0.7:
                    feedback_parts.append(
                        f"Dimensional accuracy is low ({components.dimension_accuracy:.2%}). "
                        "Check measurements against requirements."
                    )
            else:
                feedback_parts.append("Failed to measure dimensions")

        # 3. Visual quality evaluation (if enabled and code executes)
        if self.use_visual_eval and self.client:
            try:
                workplane = self.executor.execute(code)
                if workplane is not None:
                    renders = self.renderer.render_multiview(workplane, render_views)
                    visual_score, visual_feedback = self._evaluate_visual_quality(
                        renders, prompt
                    )
                    components.visual_quality = visual_score
                    feedback_parts.append(visual_feedback)
            except Exception as e:
                feedback_parts.append(f"Visual evaluation failed: {str(e)}")

        # 4. Generate comprehensive feedback
        if validation.warnings:
            feedback_parts.append("Warnings: " + "; ".join(validation.warnings))

        if validation.errors:
            feedback_parts.append("Errors: " + "; ".join(validation.errors))

        feedback = " ".join(feedback_parts) if feedback_parts else "Model looks good!"

        # 5. Calculate total reward
        total_reward = self._calculate_total_reward(components)

        return RewardResult(
            total_reward=total_reward,
            components=components,
            feedback=feedback,
            validation=validation,
            success=validation.valid,
            metadata={
                'properties': validation.properties,
                'measurements': validation.measurements
            }
        )

    def _calculate_total_reward(self, components: RewardComponents) -> float:
        """
        Calculate weighted total reward from components.

        Args:
            components: Individual reward components

        Returns:
            Total reward score (0.0 to 1.0)
        """
        # Code must be valid and execute for any positive reward
        if components.code_valid < 0.5 or components.execution_valid < 0.5:
            return 0.0

        total = (
            self.weights['code_valid'] * components.code_valid +
            self.weights['dimension_accuracy'] * components.dimension_accuracy +
            self.weights['visual_quality'] * components.visual_quality +
            self.weights['topology_valid'] * components.topology_valid
        )

        return np.clip(total, 0.0, 1.0)

    def _evaluate_visual_quality(
        self,
        renders: Dict[str, Image.Image],
        prompt: str
    ) -> tuple[float, str]:
        """
        Evaluate visual quality using vision-language model.

        Args:
            renders: Dictionary of rendered views
            prompt: Original prompt describing desired model

        Returns:
            Tuple of (score, feedback)
        """
        if not renders:
            return 0.0, "No renders available for evaluation"

        try:
            # Convert images to base64 for API
            image_contents = []
            for view_name, img in renders.items():
                # Convert PIL Image to base64
                base64_img = self.renderer.image_to_base64(img)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}"
                    }
                })

            # Construct evaluation prompt
            eval_prompt = f"""You are evaluating a 3D CAD model based on this description:
"{prompt}"

You are shown multiple views of the generated model. Please evaluate:
1. Does it match the description?
2. Are proportions realistic and appropriate?
3. Is the geometry clean and well-formed?
4. Are there any obvious defects or issues?

Provide:
- A score from 0.0 to 1.0 (where 1.0 is perfect)
- Brief feedback on strengths and areas for improvement

Format your response as:
SCORE: <number>
FEEDBACK: <text>"""

            # Call vision model
            # Note: GPT-5 models have different parameter requirements
            api_params = {
                "model": self.visual_model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": eval_prompt},
                        *image_contents
                    ]
                }]
            }

            is_gpt5 = "gpt-5" in self.visual_model.lower()
            is_gpt4_or_newer = is_gpt5 or "gpt-4" in self.visual_model.lower()

            # GPT-5 only supports temperature=1.0 (default), so don't send it
            if not is_gpt5:
                api_params["temperature"] = 0.0

            # Use max_completion_tokens for GPT-5/4 models
            if is_gpt4_or_newer:
                api_params["max_completion_tokens"] = 500
            else:
                api_params["max_tokens"] = 500

            response = self.client.chat.completions.create(**api_params)

            response_text = response.choices[0].message.content

            # Parse score and feedback
            score = 0.5  # Default
            feedback = response_text

            if "SCORE:" in response_text:
                try:
                    score_line = [line for line in response_text.split('\n') if 'SCORE:' in line][0]
                    score = float(score_line.split('SCORE:')[1].strip())
                    score = np.clip(score, 0.0, 1.0)
                except:
                    pass

            if "FEEDBACK:" in response_text:
                try:
                    feedback = response_text.split('FEEDBACK:')[1].strip()
                except:
                    pass

            return score, feedback

        except Exception as e:
            return 0.5, f"Visual evaluation error: {str(e)}"

    def _generate_failure_feedback(self, validation: ValidationResult) -> str:
        """
        Generate feedback for failed validations.

        Args:
            validation: Validation result

        Returns:
            Feedback string
        """
        parts = []

        if not validation.code_valid:
            parts.append("Code has syntax errors.")
            if validation.errors:
                parts.append("Errors: " + "; ".join(validation.errors[:3]))

        if not validation.execution_valid:
            parts.append("Code failed to execute or produce a valid CADQuery object.")

        if not validation.geometry_valid:
            parts.append("Generated geometry is invalid.")
            if validation.errors:
                parts.append("Issues: " + "; ".join(validation.errors[:3]))

        return " ".join(parts)

    def evaluate_batch(
        self,
        codes: List[str],
        prompts: List[str],
        expected_dimensions_list: Optional[List[Dict[str, float]]] = None
    ) -> List[RewardResult]:
        """
        Evaluate multiple code samples in batch.

        Args:
            codes: List of generated codes
            prompts: List of original prompts
            expected_dimensions_list: List of expected dimensions

        Returns:
            List of RewardResults
        """
        if expected_dimensions_list is None:
            expected_dimensions_list = [None] * len(codes)

        results = []
        for code, prompt, expected_dims in zip(codes, prompts, expected_dimensions_list):
            result = self.evaluate(code, prompt, expected_dims)
            results.append(result)

        return results


class RewardModelTrainer:
    """
    Trainer for a custom reward model (future enhancement).

    Instead of using GPT-5-Pro for every evaluation, we could train
    a specialized model on CAD-specific quality metrics.
    """

    def __init__(self):
        """Initialize trainer."""
        # TODO: Implement custom reward model training
        # Could use a vision transformer trained on:
        # - Expert annotations of CAD quality
        # - Geometric feature matching
        # - Dimensional accuracy prediction
        pass

    def train(self, dataset):
        """Train custom reward model."""
        raise NotImplementedError("Custom reward model training not yet implemented")

    def evaluate(self, images, code):
        """Evaluate using custom model."""
        raise NotImplementedError("Custom reward model evaluation not yet implemented")
