"""
Dataset Loading Functions.

Provides standardized dataset loading for benchmarks.

Author: Vladyslav Dehtiarov
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
import warnings


def load_iris() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the Iris dataset.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    from sklearn.datasets import load_iris
    
    data = load_iris()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names, class_names


def load_wine() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the Wine dataset.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    from sklearn.datasets import load_wine
    
    data = load_wine()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names, class_names


def load_breast_cancer() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the Breast Cancer Wisconsin dataset.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names, class_names


def load_adult() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the Adult (Census Income) dataset.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    try:
        from sklearn.datasets import fetch_openml
        
        data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        df = data.frame
        
        # Process categorical features
        X = df.drop('income', axis=1)
        y = (df['income'] == '>50K').astype(int).values
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        
        feature_names = list(X.columns)
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
        
        X = X_encoded.values.astype(float)
        
        class_names = ['<=50K', '>50K']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_names, class_names
        
    except Exception as e:
        warnings.warn(f"Could not load Adult dataset: {e}. Using synthetic data.")
        return load_synthetic_complex()


def load_german_credit() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the German Credit dataset.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    try:
        from sklearn.datasets import fetch_openml
        
        data = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
        df = data.frame
        
        # Target: 'good' = 1, 'bad' = 0
        y = (df['class'] == 'good').astype(int).values
        X = df.drop('class', axis=1)
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        
        feature_names = list(X.columns)
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
        
        X = X_encoded.values.astype(float)
        
        class_names = ['bad', 'good']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_names, class_names
        
    except Exception as e:
        warnings.warn(f"Could not load German Credit dataset: {e}. Using synthetic data.")
        return load_synthetic_complex()


def load_credit_default() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load a Credit Default dataset (synthetic if UCI unavailable).
    
    This is a domain-relevant dataset for financial applications.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    try:
        from sklearn.datasets import fetch_openml
        
        # Try to load Taiwan Credit dataset
        data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
        df = data.frame
        
        # Target column
        target_col = df.columns[-1]
        y = df[target_col].values.astype(int)
        X = df.drop(target_col, axis=1).values.astype(float)
        
        feature_names = list(df.columns[:-1])
        class_names = ['no_default', 'default']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_names, class_names
        
    except Exception:
        # Create synthetic credit default data
        np.random.seed(42)
        n_samples = 1000
        
        feature_names = [
            'credit_limit', 'age', 'education_level', 'marriage_status',
            'payment_delay_1', 'payment_delay_2', 'payment_delay_3',
            'bill_amount_1', 'bill_amount_2', 'bill_amount_3',
            'payment_amount_1', 'payment_amount_2', 'payment_amount_3'
        ]
        
        X = np.random.randn(n_samples, len(feature_names))
        
        # Simulate realistic relationships
        X[:, 0] = np.abs(X[:, 0]) * 50000 + 10000  # credit_limit
        X[:, 1] = np.abs(X[:, 1]) * 20 + 20  # age
        X[:, 2] = np.random.randint(1, 5, n_samples)  # education
        X[:, 3] = np.random.randint(1, 4, n_samples)  # marriage
        X[:, 4:7] = np.clip(X[:, 4:7] * 3, -2, 8)  # payment delays
        X[:, 7:10] = np.abs(X[:, 7:10]) * 10000  # bill amounts
        X[:, 10:13] = np.abs(X[:, 10:13]) * 5000  # payment amounts
        
        # Create target based on features
        logits = (
            -0.5 * (X[:, 0] / 50000)  # lower credit limit = higher risk
            + 0.3 * np.mean(X[:, 4:7], axis=1)  # payment delays
            - 0.2 * (X[:, 1] / 40)  # older = lower risk
            + np.random.randn(n_samples) * 0.3
        )
        y = (logits > 0).astype(int)
        
        class_names = ['no_default', 'default']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_names, class_names


def load_synthetic_simple(
    n_samples: int = 500,
    n_features: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Generate a simple synthetic dataset.
    
    The decision boundary is a simple threshold on one or two features.
    Good for testing basic algorithm correctness.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # Simple rule: y = 1 if feature_0 > 0 AND feature_1 > 0
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    class_names = ['class_0', 'class_1']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names, class_names


def load_synthetic_complex(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Generate a complex synthetic dataset.
    
    Uses sklearn's make_classification with multiple informative features.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=random_state
    )
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    class_names = ['class_0', 'class_1']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names, class_names


def load_student_performance(
    subset: str = "merged"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the UCI Student Performance dataset (Cortez & Silva, 2008).

    Predicts whether a secondary-school student passes the final grade G3
    (pass = G3 >= 10) from 30 demographic, social, and school-related features.

    Args:
        subset: one of "mat" (Mathematics, 395 students),
            "por" (Portuguese, 649 students), or "merged"
            (concatenation of both, 1044 rows).

    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    import os
    import pandas as pd

    here = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.normpath(os.path.join(here, "..", "..", "docs"))

    if subset == "mat":
        files = ["student-mat.csv"]
    elif subset == "por":
        files = ["student-por.csv"]
    elif subset == "merged":
        files = ["student-mat.csv", "student-por.csv"]
    else:
        raise ValueError(f"Unknown subset: {subset}")

    frames = [pd.read_csv(os.path.join(docs_dir, f), sep=";") for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Target: pass if final grade G3 >= 10 (Portuguese system pass threshold)
    y = (df["G3"] >= 10).astype(int).values

    # Drop grade columns to avoid target leakage; G1 and G2 trivially predict G3.
    X_df = df.drop(columns=["G1", "G2", "G3"])

    from sklearn.preprocessing import LabelEncoder

    feature_names = list(X_df.columns)
    X_encoded = X_df.copy()
    for col in X_df.columns:
        if X_df[col].dtype == "object":
            X_encoded[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X = X_encoded.values.astype(float)
    class_names = ["fail", "pass"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names, class_names


# Registry of all datasets
DATASETS = {
    'iris': load_iris,
    'wine': load_wine,
    'breast_cancer': load_breast_cancer,
    'adult': load_adult,
    'german_credit': load_german_credit,
    'credit_default': load_credit_default,
    'student_performance': load_student_performance,
    'synthetic_simple': load_synthetic_simple,
    'synthetic_complex': load_synthetic_complex,
}


def get_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Get a dataset by name.
    
    Args:
        name: Dataset name (see list_datasets())
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, class_names
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    
    return DATASETS[name]()


def list_datasets() -> List[str]:
    """
    List all available datasets.
    
    Returns:
        List of dataset names
    """
    return list(DATASETS.keys())

