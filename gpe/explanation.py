"""
Explanation data structures for GPE Framework.

This module defines the core data structures used to represent explanations:
- Condition: A single feature condition (e.g., "age > 30")
- Rule: A conjunction of conditions
- Explanation: Complete explanation with metadata

Author: Vladyslav Dehtiarov
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Dict, Any, Union
import numpy as np


def _round_down_to_float32(value: float) -> np.float32:
    """Largest float32 that does not exceed value."""
    rounded = np.float32(value)
    if float(rounded) > value:
        rounded = np.nextafter(rounded, np.float32(-np.inf))
    return rounded


@dataclass
class Condition:
    """
    Represents a single condition in a rule.
    
    A condition has the form: feature_name operator threshold
    Example: "age > 30", "income <= 50000"
    
    Attributes:
        feature_index: Index of the feature in the dataset
        feature_name: Human-readable name of the feature
        operator: Comparison operator ('<=', '>', '==', '!=', 'in')
        threshold: Value to compare against
        is_categorical: Whether this is a categorical feature
        float32_input: Compare the feature value after rounding it to float32.
            sklearn trees do this before every split test, so conditions taken
            from a tree need it to select exactly the rows the tree sends there.
    """
    feature_index: int
    feature_name: str
    operator: str  # '<=', '>', '==', '!=', 'in'
    threshold: Union[float, int, str, List]
    is_categorical: bool = False
    float32_input: bool = False
    
    def __post_init__(self):
        """Validate the condition."""
        valid_operators = ['<=', '>', '==', '!=', 'in', '>=', '<']
        if self.operator not in valid_operators:
            raise ValueError(f"Invalid operator: {self.operator}. Must be one of {valid_operators}")
    
    def evaluate(self, x: np.ndarray) -> bool:
        """
        Evaluate whether an instance satisfies this condition.
        
        Args:
            x: Feature vector (1D array)
            
        Returns:
            True if the condition is satisfied, False otherwise
        """
        value = x[self.feature_index]
        threshold = self.threshold
        if self.float32_input:
            value = float(np.float32(value))
            threshold = float(threshold)
        
        if self.operator == '<=':
            return value <= threshold
        elif self.operator == '<':
            return value < threshold
        elif self.operator == '>':
            return value > threshold
        elif self.operator == '>=':
            return value >= threshold
        elif self.operator == '==':
            return value == self.threshold
        elif self.operator == '!=':
            return value != self.threshold
        elif self.operator == 'in':
            return value in self.threshold
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.is_categorical:
            if self.operator == 'in':
                return f"{self.feature_name} in {self.threshold}"
            return f"{self.feature_name} {self.operator} '{self.threshold}'"
        else:
            if isinstance(self.threshold, float):
                return f"{self.feature_name} {self.operator} {self.threshold:.3f}"
            return f"{self.feature_name} {self.operator} {self.threshold}"
    
    def __repr__(self) -> str:
        return f"Condition({self.feature_name} {self.operator} {self.threshold})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feature_index': self.feature_index,
            'feature_name': self.feature_name,
            'operator': self.operator,
            'threshold': self.threshold,
            'is_categorical': self.is_categorical,
            'float32_input': self.float32_input
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Condition':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class Rule:
    """
    Represents a rule as a conjunction of conditions.
    
    A rule is satisfied if ALL conditions are satisfied (logical AND).
    Example: "age > 30 AND income > 50000 AND credit_score > 700"
    
    Attributes:
        conditions: List of conditions that make up the rule
        prediction: The predicted class when this rule is satisfied
        confidence: Confidence/probability of the prediction
    """
    conditions: List[Condition] = field(default_factory=list)
    prediction: Optional[Any] = None
    confidence: Optional[float] = None
    
    def add_condition(self, condition: Condition):
        """Add a condition to the rule."""
        self.conditions.append(condition)
    
    def remove_condition(self, index: int) -> Condition:
        """Remove and return the condition at the given index."""
        return self.conditions.pop(index)
    
    def evaluate(self, x: np.ndarray) -> bool:
        """
        Evaluate whether an instance satisfies this rule.
        
        Args:
            x: Feature vector (1D array)
            
        Returns:
            True if ALL conditions are satisfied, False otherwise
        """
        return all(cond.evaluate(x) for cond in self.conditions)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the rule on multiple instances.
        
        Args:
            X: Feature matrix (2D array, shape: n_samples x n_features)
            
        Returns:
            Boolean array of shape (n_samples,)
        """
        if len(self.conditions) == 0:
            return np.ones(X.shape[0], dtype=bool)
        
        result = np.ones(X.shape[0], dtype=bool)
        for cond in self.conditions:
            values = X[:, cond.feature_index]
            threshold = cond.threshold
            if cond.float32_input:
                if cond.operator in ('<=', '>'):
                    # For float32 v and float64 t, v <= t iff v <= (t rounded
                    # down to float32), which needs no float64 copy of the column.
                    values = values.astype(np.float32, copy=False)
                    threshold = _round_down_to_float32(threshold)
                else:
                    values = values.astype(np.float32).astype(np.float64)
                    threshold = float(threshold)
            
            if cond.operator == '<=':
                mask = values <= threshold
            elif cond.operator == '<':
                mask = values < threshold
            elif cond.operator == '>':
                mask = values > threshold
            elif cond.operator == '>=':
                mask = values >= threshold
            elif cond.operator == '==':
                mask = values == cond.threshold
            elif cond.operator == '!=':
                mask = values != cond.threshold
            elif cond.operator == 'in':
                mask = np.isin(values, cond.threshold)
            else:
                raise ValueError(f"Unknown operator: {cond.operator}")
            
            result = result & mask
        
        return result
    
    @property
    def complexity(self) -> int:
        """Return the number of conditions (complexity of the rule)."""
        return len(self.conditions)
    
    @property
    def features_used(self) -> List[int]:
        """Return list of feature indices used in this rule."""
        return [cond.feature_index for cond in self.conditions]
    
    @property
    def feature_names_used(self) -> List[str]:
        """Return list of feature names used in this rule."""
        return [cond.feature_name for cond in self.conditions]
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self.conditions:
            return "TRUE (no conditions)"
        
        rule_str = " AND ".join(str(cond) for cond in self.conditions)
        if self.prediction is not None:
            rule_str += f" → {self.prediction}"
            if self.confidence is not None:
                rule_str += f" (confidence: {self.confidence:.2%})"
        return rule_str
    
    def __repr__(self) -> str:
        return f"Rule({len(self.conditions)} conditions)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'conditions': [c.to_dict() for c in self.conditions],
            'prediction': self.prediction,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Rule':
        """Create from dictionary."""
        conditions = [Condition.from_dict(c) for c in d['conditions']]
        return cls(
            conditions=conditions,
            prediction=d.get('prediction'),
            confidence=d.get('confidence')
        )


@dataclass
class Certificate:
    """
    Statistical certificate of a rule produced by CertifiedGPE.
    
    Attributes:
        K: Calibration rows in the rule region that the model assigns to the class
        n: Calibration rows in the rule region
        lower_bound: Lower confidence bound on the rule's population precision.
            1.0 for the full path, whose region lies inside one leaf.
        alpha: Level of the bound; None for the full path, which needs no test
        delta: Total error budget
        min_precision: Target precision tau
        interval: 'clopper_pearson', 'hoeffding' or 'wilson'
        alpha_scheme: 'uniform', 'layered' or 'chain'
        search: 'exact', 'greedy', 'chain' or 'none'
        family_size: Number of hypotheses on this path (alpha shows the level used)
        trivial: True when the rule is the full path
        guarantee: False for 'wilson', whose coverage holds only asymptotically
        base_rate: Share of calibration rows the model assigns to the class
        min_lift: Required additive lift gamma, if any
        base_rate_upper: Upper confidence bound on the class share (lift mode)
        lift_lower_bound: lower_bound - base_rate_upper (lift mode)
        meets_lift: Whether lift_lower_bound >= min_lift (lift mode)
    """
    K: int
    n: int
    lower_bound: float
    alpha: Optional[float]
    delta: float
    min_precision: float
    interval: str
    alpha_scheme: str
    search: str
    family_size: int
    trivial: bool
    guarantee: bool
    base_rate: float = 0.0
    min_lift: Optional[float] = None
    base_rate_upper: Optional[float] = None
    lift_lower_bound: Optional[float] = None
    meets_lift: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Certificate':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class Explanation:
    """
    Complete explanation for a single instance.
    
    Contains the rule, metrics, and metadata about the explanation.
    
    Attributes:
        instance: The instance being explained
        rule: The explanation rule
        prediction: Model's prediction for the instance
        precision: Precision of the explanation
        coverage: Coverage of the explanation
        original_rule: The full rule before pruning (for comparison)
        method: Name of the method used to generate the explanation
        computation_time: Time taken to generate the explanation (seconds)
        metadata: Additional metadata
        certificate: Statistical certificate (CertifiedGPE only)
    """
    instance: np.ndarray
    rule: Rule
    prediction: Any
    precision: float = 0.0
    coverage: float = 0.0
    original_rule: Optional[Rule] = None
    method: str = "GPE"
    computation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    certificate: Optional[Certificate] = None
    
    @property
    def complexity(self) -> int:
        """Return the complexity (number of conditions) of the explanation."""
        return self.rule.complexity
    
    @property
    def reduction_ratio(self) -> float:
        """
        Return the reduction ratio compared to the original rule.
        
        Returns 0 if no original rule is available.
        """
        if self.original_rule is None:
            return 0.0
        if self.original_rule.complexity == 0:
            return 0.0
        return 1.0 - (self.rule.complexity / self.original_rule.complexity)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            "=" * 60,
            f"GPE Explanation (method: {self.method})",
            "=" * 60,
            f"Prediction: {self.prediction}",
            f"Rule: {self.rule}",
            "-" * 60,
            f"Precision: {self.precision:.2%}",
            f"Coverage: {self.coverage:.2%}",
            f"Complexity: {self.complexity} conditions",
        ]
        
        if self.original_rule is not None:
            lines.append(f"Reduction: {self.reduction_ratio:.1%} "
                        f"({self.original_rule.complexity} → {self.rule.complexity} conditions)")
        
        if self.computation_time > 0:
            lines.append(f"Computation time: {self.computation_time*1000:.1f} ms")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Explanation(prediction={self.prediction}, complexity={self.complexity}, precision={self.precision:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'instance': self.instance.tolist() if isinstance(self.instance, np.ndarray) else self.instance,
            'rule': self.rule.to_dict(),
            'prediction': self.prediction,
            'precision': self.precision,
            'coverage': self.coverage,
            'original_rule': self.original_rule.to_dict() if self.original_rule else None,
            'method': self.method,
            'computation_time': self.computation_time,
            'metadata': self.metadata,
            'certificate': self.certificate.to_dict() if self.certificate else None
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Explanation':
        """Create from dictionary."""
        return cls(
            instance=np.array(d['instance']),
            rule=Rule.from_dict(d['rule']),
            prediction=d['prediction'],
            precision=d.get('precision', 0.0),
            coverage=d.get('coverage', 0.0),
            original_rule=Rule.from_dict(d['original_rule']) if d.get('original_rule') else None,
            method=d.get('method', 'GPE'),
            computation_time=d.get('computation_time', 0.0),
            metadata=d.get('metadata', {}),
            certificate=Certificate.from_dict(d['certificate']) if d.get('certificate') else None
        )
    
    def to_natural_language(self) -> str:
        """
        Convert the explanation to natural language.
        
        Returns:
            A human-friendly description of the explanation.
        """
        if not self.rule.conditions:
            return f"The model predicts '{self.prediction}' for this instance without any specific conditions."
        
        conditions_text = []
        for cond in self.rule.conditions:
            if cond.operator == '<=':
                conditions_text.append(f"{cond.feature_name} is at most {cond.threshold:.2f}")
            elif cond.operator == '<':
                conditions_text.append(f"{cond.feature_name} is less than {cond.threshold:.2f}")
            elif cond.operator == '>':
                conditions_text.append(f"{cond.feature_name} is greater than {cond.threshold:.2f}")
            elif cond.operator == '>=':
                conditions_text.append(f"{cond.feature_name} is at least {cond.threshold:.2f}")
            elif cond.operator == '==':
                conditions_text.append(f"{cond.feature_name} equals {cond.threshold}")
            elif cond.operator == '!=':
                conditions_text.append(f"{cond.feature_name} is not {cond.threshold}")
            elif cond.operator == 'in':
                conditions_text.append(f"{cond.feature_name} is one of {cond.threshold}")
        
        if len(conditions_text) == 1:
            conditions_str = conditions_text[0]
        elif len(conditions_text) == 2:
            conditions_str = f"{conditions_text[0]} and {conditions_text[1]}"
        else:
            conditions_str = ", ".join(conditions_text[:-1]) + f", and {conditions_text[-1]}"
        
        return (f"The model predicts '{self.prediction}' because {conditions_str}. "
                f"This explanation covers {self.coverage:.1%} of similar cases "
                f"with {self.precision:.1%} accuracy.")

