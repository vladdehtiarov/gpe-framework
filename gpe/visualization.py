"""
Visualization Tools for GPE Explanations.

This module provides functions to visualize explanations:
- Text-based explanations
- HTML explanations
- Matplotlib plots
- Comparison charts

Author: Vladyslav Dehtiarov
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import warnings

from .explanation import Explanation, Rule, Condition


def explanation_to_text(
    explanation: Explanation,
    show_metrics: bool = True,
    show_instance: bool = False
) -> str:
    """
    Convert an explanation to a formatted text string.
    
    Args:
        explanation: The explanation to format
        show_metrics: Whether to include precision/coverage/complexity
        show_instance: Whether to show the original instance values
        
    Returns:
        Formatted text string
    """
    lines = []
    
    lines.append("=" * 60)
    lines.append(f"GPE EXPLANATION ({explanation.method})")
    lines.append("=" * 60)
    
    # Prediction
    lines.append(f"\n📊 Prediction: {explanation.prediction}")
    
    # Instance values
    if show_instance:
        lines.append("\n📋 Instance:")
        for i, cond in enumerate(explanation.rule.conditions):
            value = explanation.instance[cond.feature_index]
            lines.append(f"   {cond.feature_name}: {value:.4f}")
    
    # Rule
    lines.append("\n📜 Explanation Rule:")
    if len(explanation.rule.conditions) == 0:
        lines.append("   No specific conditions (always true)")
    else:
        for i, cond in enumerate(explanation.rule.conditions, 1):
            lines.append(f"   {i}. {cond}")
    
    # Metrics
    if show_metrics:
        lines.append("\n📈 Metrics:")
        lines.append(f"   Precision: {explanation.precision:.1%}")
        lines.append(f"   Coverage:  {explanation.coverage:.1%}")
        lines.append(f"   Complexity: {explanation.complexity} conditions")
        
        if explanation.original_rule is not None:
            reduction = explanation.reduction_ratio
            lines.append(f"   Reduction: {reduction:.1%} "
                        f"({explanation.original_rule.complexity} → {explanation.complexity})")
        
        if explanation.computation_time > 0:
            lines.append(f"   Time: {explanation.computation_time*1000:.1f} ms")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def explanation_to_html(
    explanation: Explanation,
    show_metrics: bool = True,
    style: str = 'default'
) -> str:
    """
    Convert an explanation to HTML format.
    
    Args:
        explanation: The explanation to format
        show_metrics: Whether to include metrics
        style: CSS style preset ('default', 'dark', 'minimal')
        
    Returns:
        HTML string
    """
    # CSS styles
    styles = {
        'default': """
            .gpe-explanation {
                font-family: 'Segoe UI', Arial, sans-serif;
                max-width: 600px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
            .gpe-header {
                font-size: 18px;
                font-weight: bold;
                color: #212529;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .gpe-prediction {
                font-size: 24px;
                font-weight: bold;
                color: #28a745;
                margin: 10px 0;
            }
            .gpe-rule {
                background: white;
                padding: 15px;
                border-radius: 4px;
                margin: 10px 0;
            }
            .gpe-condition {
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }
            .gpe-condition:last-child {
                border-bottom: none;
            }
            .gpe-feature {
                font-weight: bold;
                color: #007bff;
            }
            .gpe-operator {
                color: #6c757d;
            }
            .gpe-threshold {
                font-weight: bold;
                color: #dc3545;
            }
            .gpe-metrics {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin-top: 15px;
            }
            .gpe-metric {
                text-align: center;
                padding: 10px;
                background: white;
                border-radius: 4px;
            }
            .gpe-metric-value {
                font-size: 20px;
                font-weight: bold;
                color: #212529;
            }
            .gpe-metric-label {
                font-size: 12px;
                color: #6c757d;
            }
        """,
        'dark': """
            .gpe-explanation {
                font-family: 'Fira Code', monospace;
                max-width: 600px;
                padding: 20px;
                background: #1e1e1e;
                border-radius: 8px;
                color: #d4d4d4;
            }
            .gpe-header {
                font-size: 18px;
                color: #569cd6;
                border-bottom: 2px solid #569cd6;
                padding-bottom: 10px;
            }
            .gpe-prediction {
                font-size: 24px;
                color: #4ec9b0;
            }
            .gpe-rule {
                background: #2d2d2d;
                padding: 15px;
                border-radius: 4px;
            }
            .gpe-feature { color: #9cdcfe; }
            .gpe-operator { color: #d4d4d4; }
            .gpe-threshold { color: #ce9178; }
            .gpe-metric-value { color: #dcdcaa; }
        """,
        'minimal': """
            .gpe-explanation {
                font-family: system-ui;
                max-width: 500px;
                padding: 15px;
            }
            .gpe-header {
                font-size: 16px;
                font-weight: bold;
            }
            .gpe-prediction {
                font-size: 20px;
                color: #2563eb;
            }
        """
    }
    
    css = styles.get(style, styles['default'])
    
    # Build HTML
    html_parts = [f"<style>{css}</style>"]
    html_parts.append('<div class="gpe-explanation">')
    
    # Header
    html_parts.append(f'<div class="gpe-header">GPE Explanation ({explanation.method})</div>')
    
    # Prediction
    html_parts.append(f'<div class="gpe-prediction">Prediction: {explanation.prediction}</div>')
    
    # Rule
    html_parts.append('<div class="gpe-rule">')
    if len(explanation.rule.conditions) == 0:
        html_parts.append('<p>No specific conditions</p>')
    else:
        for cond in explanation.rule.conditions:
            threshold_str = f"{cond.threshold:.3f}" if isinstance(cond.threshold, float) else str(cond.threshold)
            html_parts.append(
                f'<div class="gpe-condition">'
                f'<span class="gpe-feature">{cond.feature_name}</span> '
                f'<span class="gpe-operator">{cond.operator}</span> '
                f'<span class="gpe-threshold">{threshold_str}</span>'
                f'</div>'
            )
    html_parts.append('</div>')
    
    # Metrics
    if show_metrics:
        html_parts.append('<div class="gpe-metrics">')
        
        html_parts.append(
            f'<div class="gpe-metric">'
            f'<div class="gpe-metric-value">{explanation.precision:.1%}</div>'
            f'<div class="gpe-metric-label">Precision</div>'
            f'</div>'
        )
        
        html_parts.append(
            f'<div class="gpe-metric">'
            f'<div class="gpe-metric-value">{explanation.coverage:.1%}</div>'
            f'<div class="gpe-metric-label">Coverage</div>'
            f'</div>'
        )
        
        html_parts.append(
            f'<div class="gpe-metric">'
            f'<div class="gpe-metric-value">{explanation.complexity}</div>'
            f'<div class="gpe-metric-label">Conditions</div>'
            f'</div>'
        )
        
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    
    return "\n".join(html_parts)


def plot_explanation(
    explanation: Explanation,
    instance_values: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Any:
    """
    Create a visual plot of the explanation.
    
    Shows each condition as a bar with the threshold and instance value.
    
    Args:
        explanation: The explanation to plot
        instance_values: Dict mapping feature names to their values
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot create plot.")
        return None
    
    conditions = explanation.rule.conditions
    
    if len(conditions) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No conditions to display", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get instance values
    if instance_values is None:
        instance_values = {}
        for cond in conditions:
            instance_values[cond.feature_name] = explanation.instance[cond.feature_index]
    
    # Plot each condition
    y_positions = list(range(len(conditions)))
    
    for i, cond in enumerate(conditions):
        feature_name = cond.feature_name
        threshold = cond.threshold
        instance_val = instance_values.get(feature_name, 0)
        
        # Determine color based on operator
        if cond.operator in ['<=', '<']:
            color = '#3498db'  # Blue
            label = f"{feature_name} {cond.operator} {threshold:.2f}"
        else:
            color = '#e74c3c'  # Red
            label = f"{feature_name} {cond.operator} {threshold:.2f}"
        
        # Draw threshold line
        ax.barh(i, threshold, color=color, alpha=0.3, height=0.5)
        
        # Draw instance value marker
        ax.scatter([instance_val], [i], color='black', s=100, zorder=5, marker='o')
        
        # Add text
        ax.text(threshold, i + 0.3, f"threshold: {threshold:.2f}", fontsize=9)
        ax.text(instance_val, i - 0.3, f"value: {instance_val:.2f}", fontsize=9)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([c.feature_name for c in conditions])
    ax.set_xlabel("Feature Value")
    ax.set_title(f"GPE Explanation for prediction: {explanation.prediction}\n"
                f"Precision: {explanation.precision:.1%}, Coverage: {explanation.coverage:.1%}")
    
    # Legend
    blue_patch = mpatches.Patch(color='#3498db', alpha=0.3, label='≤ / < threshold')
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.3, label='> / ≥ threshold')
    ax.legend(handles=[blue_patch, red_patch], loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison(
    explanations: Dict[str, Explanation],
    metric: str = 'complexity',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Any:
    """
    Compare explanations from different methods.
    
    Args:
        explanations: Dict mapping method names to explanations
        metric: Metric to compare ('complexity', 'precision', 'coverage', 'time')
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot create plot.")
        return None
    
    methods = list(explanations.keys())
    
    if metric == 'complexity':
        values = [explanations[m].complexity for m in methods]
        ylabel = "Number of Conditions"
        title = "Explanation Complexity Comparison"
    elif metric == 'precision':
        values = [explanations[m].precision for m in methods]
        ylabel = "Precision"
        title = "Explanation Precision Comparison"
    elif metric == 'coverage':
        values = [explanations[m].coverage for m in methods]
        ylabel = "Coverage"
        title = "Explanation Coverage Comparison"
    elif metric == 'time':
        values = [explanations[m].computation_time * 1000 for m in methods]
        ylabel = "Computation Time (ms)"
        title = "Computation Time Comparison"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, values, color=colors)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}' if metric != 'complexity' else f'{int(value)}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_summary(
    explanations: List[Explanation],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> Any:
    """
    Plot summary metrics for a list of explanations.
    
    Creates a multi-panel plot showing distributions of:
    - Precision
    - Coverage  
    - Complexity
    - Computation Time
    
    Args:
        explanations: List of explanations to summarize
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot create plot.")
        return None
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Precision
    precisions = [e.precision for e in explanations]
    axes[0].hist(precisions, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(precisions), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(precisions):.2f}')
    axes[0].set_xlabel('Precision')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Precision Distribution')
    axes[0].legend()
    
    # Coverage
    coverages = [e.coverage for e in explanations]
    axes[1].hist(coverages, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(coverages), color='red', linestyle='--',
                    label=f'Mean: {np.mean(coverages):.2f}')
    axes[1].set_xlabel('Coverage')
    axes[1].set_title('Coverage Distribution')
    axes[1].legend()
    
    # Complexity
    complexities = [e.complexity for e in explanations]
    axes[2].hist(complexities, bins=range(max(complexities)+2), 
                 color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[2].axvline(np.mean(complexities), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(complexities):.1f}')
    axes[2].set_xlabel('# Conditions')
    axes[2].set_title('Complexity Distribution')
    axes[2].legend()
    
    # Time
    times = [e.computation_time * 1000 for e in explanations]
    axes[3].hist(times, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[3].axvline(np.mean(times), color='red', linestyle='--',
                    label=f'Mean: {np.mean(times):.1f}ms')
    axes[3].set_xlabel('Time (ms)')
    axes[3].set_title('Computation Time')
    axes[3].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def print_explanation(explanation: Explanation, **kwargs):
    """
    Print an explanation to the console.
    
    Convenience function that prints the text representation.
    
    Args:
        explanation: The explanation to print
        **kwargs: Additional arguments passed to explanation_to_text
    """
    print(explanation_to_text(explanation, **kwargs))


def display_explanation_notebook(explanation: Explanation, style: str = 'default'):
    """
    Display an explanation in a Jupyter notebook.
    
    Args:
        explanation: The explanation to display
        style: CSS style preset
    """
    try:
        from IPython.display import HTML, display
        html = explanation_to_html(explanation, style=style)
        display(HTML(html))
    except ImportError:
        print(explanation_to_text(explanation))

