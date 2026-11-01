"""
Compiled explanations: a table from tree leaves to rules.

GPE-Core, GPE-IT and Certified-GPE build the rule from the literals of the path,
the predicted class and the background data, so the explanation of x is a function
of its leaf. All rules can therefore be computed offline. Explaining an instance
then costs one lookup, and needs no background data on the device.

Author: Vladyslav Dehtiarov
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .certified import CertifiedGPE
from .core import GPEExplainer
from .explanation import Certificate, Condition, Explanation, Rule
from .novel_methods import GPEInformationTheoretic
from .tree_utils import get_leaf_paths, representative_instance, simplify_conditions

_LEAF_DETERMINED = (GPEExplainer, GPEInformationTheoretic, CertifiedGPE)


@dataclass
class CompiledEntry:
    """Explanation of one leaf."""
    rule: Rule
    original_rule: Rule
    prediction: Any
    precision: float
    coverage: float
    mask: int  # bit d is set when the literal at depth d of the path is kept
    depth: int
    certificate: Optional[Certificate] = None


@dataclass
class CompiledExplanations:
    """
    Table leaf -> rule for one tree and one background set.

    Attributes:
        model: The fitted tree
        entries: Leaf node ID -> CompiledEntry
        method: Name of the method that produced the rules
        simultaneous: Whether the certificates hold for all leaves at once
    """
    model: Any
    entries: Dict[int, CompiledEntry] = field(default_factory=dict)
    method: str = "GPE"
    simultaneous: bool = False

    def explain(self, x: np.ndarray) -> Explanation:
        """Look up the explanation of an instance."""
        start_time = time.time()
        x = np.asarray(x).flatten()
        entry = self.entries[int(self.model.apply(x.reshape(1, -1))[0])]
        return Explanation(
            instance=x,
            rule=entry.rule,
            prediction=entry.prediction,
            precision=entry.precision,
            coverage=entry.coverage,
            original_rule=entry.original_rule,
            method=f"{self.method}-compiled",
            computation_time=time.time() - start_time,
            metadata={'simultaneous': self.simultaneous},
            certificate=entry.certificate,
        )

    def explain_batch(self, X: np.ndarray) -> List[Explanation]:
        """Look up explanations for multiple instances."""
        return [self.explain(x) for x in np.asarray(X)]

    @property
    def n_unique_rules(self) -> int:
        """Number of distinct rules in the table; at most the number of leaves."""
        return len({
            tuple(sorted((c.feature_index, c.operator, float(c.threshold)) for c in e.rule.conditions))
            for e in self.entries.values()
        })

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """
        Export the table for a device that already stores the tree.

        Returns:
            Arrays sorted by leaf ID: leaf_ids, masks (uint64), and calibration
            counts K and n (uint32, zero without a certificate). A set bit d in a
            mask keeps the literal at depth d of that leaf's path.
        """
        leaf_ids = np.array(sorted(self.entries), dtype=np.int64)
        depth = max((self.entries[int(leaf)].depth for leaf in leaf_ids), default=0)
        if depth > 64:
            raise ValueError(f"Tree depth {depth} does not fit a 64-bit mask")
        entries = [self.entries[int(leaf)] for leaf in leaf_ids]
        return {
            'leaf_ids': leaf_ids,
            'masks': np.array([e.mask for e in entries], dtype=np.uint64),
            'K': np.array([e.certificate.K if e.certificate else 0 for e in entries], dtype=np.uint32),
            'n': np.array([e.certificate.n if e.certificate else 0 for e in entries], dtype=np.uint32),
        }

    def storage_bytes(self, with_counts: bool = False) -> int:
        """
        Size of the mask table, at the smallest whole-byte mask width for this tree.

        Args:
            with_counts: Add the uint32 K and n per leaf, needed to recalibrate
                         on the device

        Returns:
            Bytes used on top of the tree itself
        """
        depth = max((e.depth for e in self.entries.values()), default=0)
        per_leaf = math.ceil(depth / 8) + (8 if with_counts else 0)
        return len(self.entries) * per_leaf


def compile_explanations(
    explainer: GPEExplainer,
    X_background: Optional[np.ndarray] = None,
    simultaneous: bool = False
) -> CompiledExplanations:
    """
    Compute the explanation of every leaf of the tree.

    Args:
        explainer: GPEExplainer, GPEInformationTheoretic or CertifiedGPE. Other
                   variants may use more of x than its leaf and are rejected.
        X_background: Background data; for CertifiedGPE it must be None (X_cal is used)
        simultaneous: CertifiedGPE only. Split the budget over all leaves, so that
                      the certificates hold at once for the whole tree. Otherwise
                      each entry holds on its own, with probability 1 - delta.

    Returns:
        CompiledExplanations table
    """
    if type(explainer) not in _LEAF_DETERMINED:
        raise TypeError(
            f"{type(explainer).__name__} is not known to depend on x only through the leaf; "
            f"supported: {[cls.__name__ for cls in _LEAF_DETERMINED]}"
        )
    is_certified = isinstance(explainer, CertifiedGPE)
    if is_certified and X_background is not None:
        raise ValueError("CertifiedGPE evaluates rules on X_cal; X_background is not accepted")
    if simultaneous and not is_certified:
        raise ValueError("simultaneous=True is available only for CertifiedGPE")

    model = explainer.model
    paths = get_leaf_paths(model, explainer.feature_names)

    split = {}
    if simultaneous:
        lengths = [
            len(simplify_conditions([literal for _, literal in path]) if explainer.simplify else path)
            for path in paths.values()
        ]
        split = {
            'family_total': sum(2 ** k - 2 for k in lengths if k >= 2) or None,
            'n_leaves': max(1, sum(k >= 2 for k in lengths)),
            'n_classes': len(model.classes_),
        }

    table = CompiledExplanations(model=model, simultaneous=simultaneous)
    for leaf, path in paths.items():
        x = representative_instance(model, path)
        if int(model.apply(x.reshape(1, -1))[0]) != leaf:
            raise RuntimeError(f"Could not build an instance that reaches leaf {leaf}")
        if is_certified:
            explanation = explainer._explain_leaf(x, **split)
        else:
            explanation = explainer.explain(x, X_background)
        table.method = explanation.method
        table.entries[leaf] = CompiledEntry(
            rule=explanation.rule,
            original_rule=explanation.original_rule,
            prediction=explanation.prediction,
            precision=float(explanation.precision),
            coverage=float(explanation.coverage),
            mask=_literal_mask(path, explanation.rule.conditions),
            depth=len(path),
            certificate=explanation.certificate,
        )
    return table


def _literal_mask(path, conditions: List[Condition]) -> int:
    """Encode kept literals as bits over the depth of the path."""
    mask = 0
    for cond in conditions:
        for depth, (_, literal) in enumerate(path):
            if (
                literal.feature_index == cond.feature_index
                and literal.operator == cond.operator
                and literal.threshold == cond.threshold
            ):
                mask |= 1 << depth
                break
        else:
            raise RuntimeError(f"Literal {cond!r} is not on the path")
    return mask
