"""
LLM-based CADQuery code generation with structured prompting.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    model: str = "gpt-5-2025-08-07"
    temperature: float = 0.7
    max_tokens: int = 8192  # High limit for GPT-5 extensive reasoning tokens
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class GenerationResult:
    """Result of code generation."""
    code: str
    prompt: str
    model: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CodeGenerator:
    """
    Generates CADQuery Python code from natural language descriptions.

    Uses OpenAI's latest models with specialized prompting for spatial reasoning.
    """

    SYSTEM_PROMPT = """You are an expert CAD engineer specialized in generating CadQuery Python code.

Your task is to generate valid, executable CadQuery code that creates 3D models based on natural language descriptions.

Guidelines:
1. Always import cadquery as cq
2. Create clean, well-commented code
3. Use precise dimensions and measurements
4. Ensure the final object is assigned to a variable named 'result'
5. Use proper CadQuery API methods (Workplane, box, cylinder, etc.)
6. Consider real-world proportions and constraints
7. Build models step-by-step with clear operations

Example structure:
```python
import cadquery as cq

# Create base
base = cq.Workplane("XY").box(width, depth, height)

# Add features
...

# Combine and assign to result
result = base.union(other_parts)
```

Focus on spatial reasoning and geometric precision."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize code generator.

        Args:
            api_key: OpenAI API key. If None, reads from environment
            config: Generation configuration
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=self.api_key)
        self.config = config or GenerationConfig()

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        Generate CADQuery code from a natural language prompt.

        Args:
            prompt: Natural language description of desired CAD model
            context: Additional context or requirements
            constraints: Dictionary of constraints (e.g., dimensions, materials)

        Returns:
            GenerationResult containing generated code and metadata
        """
        try:
            # Build the user prompt
            user_prompt = self._build_prompt(prompt, context, constraints)

            # Call OpenAI API
            # Note: GPT-5 models have different parameter requirements
            api_params = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            }

            is_gpt5 = "gpt-5" in self.config.model.lower()
            is_gpt4_or_newer = is_gpt5 or "gpt-4" in self.config.model.lower()

            # GPT-5 only supports temperature=1.0 (default), so don't send sampling params
            if not is_gpt5:
                api_params["temperature"] = self.config.temperature
                api_params["top_p"] = self.config.top_p
                api_params["frequency_penalty"] = self.config.frequency_penalty
                api_params["presence_penalty"] = self.config.presence_penalty

            # Use max_completion_tokens for GPT-4/5, max_tokens for older models
            if is_gpt4_or_newer:
                api_params["max_completion_tokens"] = self.config.max_tokens
            else:
                api_params["max_tokens"] = self.config.max_tokens

            response = self.client.chat.completions.create(**api_params)

            # Extract code from response
            generated_text = response.choices[0].message.content
            code = self._extract_code(generated_text)

            # Check if we hit token limit
            finish_reason = response.choices[0].finish_reason
            if finish_reason == 'length' and not code:
                # Model used all tokens for reasoning, no code output
                return GenerationResult(
                    code="",
                    prompt=user_prompt,
                    model=self.config.model,
                    success=False,
                    error=f"Token limit reached (finish_reason: {finish_reason}). "
                          f"Model used all tokens for reasoning without outputting code. "
                          f"Try increasing max_completion_tokens or using a simpler prompt.",
                    metadata={
                        'usage': response.usage.model_dump() if response.usage else None,
                        'finish_reason': finish_reason
                    }
                )

            return GenerationResult(
                code=code,
                prompt=user_prompt,
                model=self.config.model,
                success=True,
                metadata={
                    'usage': response.usage.model_dump() if response.usage else None,
                    'finish_reason': finish_reason
                }
            )

        except Exception as e:
            return GenerationResult(
                code="",
                prompt=prompt,
                model=self.config.model,
                success=False,
                error=str(e)
            )

    def _build_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build detailed prompt from inputs.

        Args:
            prompt: Base prompt
            context: Additional context
            constraints: Constraints dictionary

        Returns:
            Formatted prompt string
        """
        parts = [f"Create a 3D CAD model: {prompt}"]

        if context:
            parts.append(f"\nAdditional context: {context}")

        if constraints:
            parts.append("\nConstraints:")
            for key, value in constraints.items():
                parts.append(f"- {key}: {value}")

        parts.append("\nGenerate only the Python code with CadQuery. Ensure the final model is assigned to 'result'.")

        return "\n".join(parts)

    def _extract_code(self, text: str) -> str:
        """
        Extract Python code from LLM response.

        Args:
            text: Raw LLM response text

        Returns:
            Extracted Python code
        """
        if not text or not text.strip():
            return ""

        # Remove markdown code blocks
        if "```python" in text:
            # Extract code between ```python and ```
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end > start:
                code = text[start:end].strip()
            else:
                code = text[start:].strip()
        elif "```" in text:
            # Extract code between ``` and ```
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                code = text[start:end].strip()
            else:
                code = text[start:].strip()
        else:
            # Check if text looks like code (has import or cq in it)
            if "import" in text or "cq." in text or "cadquery" in text:
                code = text.strip()
            else:
                # Response might be explanation without code
                # Try to find any Python-like content
                lines = text.split('\n')
                code_lines = [line for line in lines if
                             'import' in line or 'cq.' in line or
                             '=' in line or 'def ' in line]
                code = '\n'.join(code_lines).strip()

        return code

    def generate_with_feedback(
        self,
        prompt: str,
        feedback: str,
        previous_code: str,
        max_iterations: int = 3
    ) -> GenerationResult:
        """
        Generate code with iterative feedback.

        Args:
            prompt: Original prompt
            feedback: Feedback on previous attempt
            previous_code: Previous generated code
            max_iterations: Maximum refinement iterations

        Returns:
            GenerationResult with refined code
        """
        refinement_prompt = f"""Original request: {prompt}

Previous code:
```python
{previous_code}
```

Feedback: {feedback}

Please generate improved CadQuery code that addresses the feedback."""

        return self.generate(refinement_prompt)

    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 3,
        temperature: Optional[float] = None
    ) -> List[GenerationResult]:
        """
        Generate multiple variations of the same prompt.

        Args:
            prompt: Natural language description
            num_variations: Number of variations to generate
            temperature: Override temperature for variation

        Returns:
            List of GenerationResults
        """
        original_temp = self.config.temperature
        if temperature is not None:
            self.config.temperature = temperature

        variations = []
        for i in range(num_variations):
            result = self.generate(prompt)
            variations.append(result)

        # Restore original temperature
        self.config.temperature = original_temp

        return variations


class PromptLibrary:
    """
    Library of template prompts for common CAD objects.
    """

    TEMPLATES = {
        'chair': {
            'base': 'Create a chair with four legs, a seat, and a backrest',
            'constraints': {
                'seat_height': '45cm from ground',
                'seat_dimensions': '40cm x 40cm',
                'backrest_height': '40cm above seat'
            }
        },
        'table': {
            'base': 'Create a table with four legs and a flat top',
            'constraints': {
                'table_height': '75cm',
                'top_dimensions': '120cm x 80cm',
                'leg_thickness': '5cm x 5cm'
            }
        },
        'box': {
            'base': 'Create a rectangular box with specified dimensions',
            'constraints': {
                'length': '10cm',
                'width': '8cm',
                'height': '5cm'
            }
        },
        'cylinder': {
            'base': 'Create a cylinder with specified radius and height',
            'constraints': {
                'radius': '5cm',
                'height': '10cm'
            }
        }
    }

    @classmethod
    def get_template(cls, object_type: str) -> Optional[Dict[str, Any]]:
        """Get template for a specific object type."""
        return cls.TEMPLATES.get(object_type.lower())

    @classmethod
    def list_templates(cls) -> List[str]:
        """List available template types."""
        return list(cls.TEMPLATES.keys())
