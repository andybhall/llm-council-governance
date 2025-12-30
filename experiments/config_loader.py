"""YAML configuration loader for experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""

    name: str
    n_questions: int = 40


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Experiment metadata
    name: str = "experiment"
    description: str = ""

    # Model configuration
    council_models: List[str] = field(default_factory=list)
    chairman_model: str = ""

    # Benchmark configuration
    benchmarks: List[BenchmarkConfig] = field(default_factory=list)

    # Structure configuration
    structures: List[str] = field(default_factory=list)

    # Settings
    n_replications: int = 3
    temperature: float = 0.0
    timeout: float = 60.0

    # Output configuration
    output_dir: str = "experiments/results"
    save_interval: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from a dictionary."""
        experiment = data.get("experiment", {})
        models = data.get("models", {})
        benchmarks_data = data.get("benchmarks", [])
        settings = data.get("settings", {})
        output = data.get("output", {})

        # Parse benchmarks
        benchmarks = []
        for b in benchmarks_data:
            if isinstance(b, str):
                benchmarks.append(BenchmarkConfig(name=b))
            else:
                benchmarks.append(
                    BenchmarkConfig(
                        name=b.get("name", ""),
                        n_questions=b.get("n_questions", 40),
                    )
                )

        return cls(
            name=experiment.get("name", "experiment"),
            description=experiment.get("description", ""),
            council_models=models.get("council", []),
            chairman_model=models.get("chairman", ""),
            benchmarks=benchmarks,
            structures=data.get("structures", []),
            n_replications=settings.get("n_replications", 3),
            temperature=settings.get("temperature", 0.0),
            timeout=settings.get("timeout", 60.0),
            output_dir=output.get("dir", "experiments/results"),
            save_interval=output.get("save_interval", 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "experiment": {
                "name": self.name,
                "description": self.description,
            },
            "models": {
                "council": self.council_models,
                "chairman": self.chairman_model,
            },
            "benchmarks": [
                {"name": b.name, "n_questions": b.n_questions}
                for b in self.benchmarks
            ],
            "structures": self.structures,
            "settings": {
                "n_replications": self.n_replications,
                "temperature": self.temperature,
                "timeout": self.timeout,
            },
            "output": {
                "dir": self.output_dir,
                "save_interval": self.save_interval,
            },
        }


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        ExperimentConfig object

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return ExperimentConfig.from_yaml(config_path)


def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate an experiment configuration.

    Args:
        config: ExperimentConfig to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not config.council_models:
        errors.append("No council models specified")

    if not config.chairman_model:
        errors.append("No chairman model specified")

    if not config.benchmarks:
        errors.append("No benchmarks specified")

    if not config.structures:
        errors.append("No structures specified")

    valid_structures = {
        "rank_synthesize",
        "majority_vote",
        "deliberate_vote",
        "deliberate_synthesize",
        "weighted_vote",
    }
    for structure in config.structures:
        if structure not in valid_structures:
            errors.append(f"Unknown structure: {structure}")

    if config.n_replications < 1:
        errors.append("n_replications must be at least 1")

    if config.temperature < 0 or config.temperature > 2:
        errors.append("temperature must be between 0 and 2")

    return errors
