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
import math
import numpy as np


def _round_down_to_float32(value: float) -> np.float32:
    """Largest float32 that does not exceed value."""
    rounded = np.float32(value)
    if float(rounded) > value:
        rounded = np.nextafter(rounded, np.float32(-np.inf))
    return rounded


def _json_safe(value: Any) -> Any:
    """
    Convert a value to what strict JSON accepts.
    
    NumPy scalars and arrays become Python values, NaN becomes None and an
    infinity becomes the string 'Infinity' or '-Infinity'.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        value = sorted(value, key=str)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
    return value


def _float_from_json(value: Any) -> float:
    """Inverse of _json_safe for a float."""
    return float('nan') if value is None else float(value)


def _percent(value: float, digits: int = 2) -> str:
    """Format a share; NaN means it could not be measured."""
    return "n/a" if value != value else f"{value:.{digits}%}"


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
        matches_missing: A missing value (NaN) satisfies this condition. sklearn
            trees route NaN down one branch of each split, so conditions taken
            from a tree need it to select the same rows as the tree.
        missing_in_instance: Display only: the explained instance reached this
            condition with a missing value, so the text says so.
    """
    feature_index: int
    feature_name: str
    operator: str  # '<=', '>', '==', '!=', 'in'
    threshold: Union[float, int, str, List]
    is_categorical: bool = False
    float32_input: bool = False
    matches_missing: bool = False
    missing_in_instance: bool = field(default=False, compare=False)
    
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
        if self.matches_missing and value != value:
            return True
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
        suffix = " (or missing)" if self.missing_in_instance else ""
        if isinstance(self.threshold, float):
            return f"{self.feature_name} {self.operator} {self.threshold:.3f}{suffix}"
        return f"{self.feature_name} {self.operator} {self.threshold}{suffix}"
    
    def __repr__(self) -> str:
        return f"Condition({self.feature_name} {self.operator} {self.threshold})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return _json_safe({
            'feature_index': self.feature_index,
            'feature_name': self.feature_name,
            'operator': self.operator,
            'threshold': self.threshold,
            'is_categorical': self.is_categorical,
            'float32_input': self.float32_input,
            'matches_missing': self.matches_missing,
            'missing_in_instance': self.missing_in_instance
        })
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Condition':
        """Create from dictionary."""
        d = dict(d)
        if not d.get('is_categorical') and d.get('threshold') in ('Infinity', '-Infinity'):
            d['threshold'] = float(d['threshold'])
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
        confidence: Purity of the leaf the rule came from, on the training data.
            It is not the precision of the rule: see Explanation.precision.
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
            
            if cond.matches_missing:
                mask = mask | np.isnan(X[:, cond.feature_index])
            
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
                rule_str += f" (leaf purity: {self.confidence:.2%})"
        return rule_str
    
    def __repr__(self) -> str:
        return f"Rule({len(self.conditions)} conditions)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'conditions': [c.to_dict() for c in self.conditions],
            'prediction': _json_safe(self.prediction),
            'confidence': _json_safe(self.confidence)
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
        return _json_safe(asdict(self))
    
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
        precision: Share of the background rows in the rule's region that the
            model assigns to the predicted class. This is agreement with the model,
            not correctness of the prediction. NaN when no rows fall in the region.
        coverage: Share of the background rows in the rule's region; NaN without
            background data
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
            f"Precision: {_percent(self.precision)}",
            f"Coverage: {_percent(self.coverage)}",
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
            'instance': _json_safe(self.instance),
            'rule': self.rule.to_dict(),
            'prediction': _json_safe(self.prediction),
            'precision': _json_safe(self.precision),
            'coverage': _json_safe(self.coverage),
            'original_rule': self.original_rule.to_dict() if self.original_rule else None,
            'method': self.method,
            'computation_time': _json_safe(self.computation_time),
            'metadata': _json_safe(self.metadata),
            'certificate': self.certificate.to_dict() if self.certificate else None
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Explanation':
        """Create from dictionary."""
        return cls(
            instance=np.array([_float_from_json(v) for v in d['instance']], dtype=np.float64),
            rule=Rule.from_dict(d['rule']),
            prediction=d['prediction'],
            precision=_float_from_json(d.get('precision', 0.0)),
            coverage=_float_from_json(d.get('coverage', 0.0)),
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
            if cond.missing_in_instance:
                conditions_text.append(f"{cond.feature_name} is missing")
            elif cond.operator == '<=':
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
        
        text = f"The model predicts '{self.prediction}' because {conditions_str}."
        if self.precision != self.precision:
            return text + " No background rows satisfy this rule, so its precision is unknown."
        return (f"{text} The rule holds for {self.coverage:.1%} of the background rows, "
                f"and the model gives {self.precision:.1%} of them the same class.")

