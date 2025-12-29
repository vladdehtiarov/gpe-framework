#!/usr/bin/env python3
"""
GPE Framework - Демонстрація
============================

Цей скрипт демонструє основні можливості GPE Framework.

Запуск:
    python demo.py
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# GPE
from gpe import GPEExplainer, GPEInformationTheoretic


def main():
    print("="*60)
    print("  GPE Framework - Демонстрація")
    print("="*60)
    
    # 1. Генерація даних (імітація кредитного скорингу)
    print("\n📊 Генерація даних...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # Ознаки
    income = np.random.normal(60000, 25000, n_samples).clip(15000, 200000)
    age = np.random.normal(35, 12, n_samples).clip(18, 70)
    debt_ratio = np.random.uniform(0.1, 0.8, n_samples)
    credit_score = np.random.normal(680, 80, n_samples).clip(300, 850)
    employment_years = np.random.exponential(5, n_samples).clip(0, 40)
    
    # Target: ймовірність дефолту
    default_prob = (
        0.3 * (income < 40000).astype(float) +
        0.3 * (debt_ratio > 0.5).astype(float) +
        0.2 * (credit_score < 600).astype(float) +
        0.1 * (employment_years < 1).astype(float) +
        0.1 * (age < 25).astype(float)
    )
    y = (np.random.random(n_samples) < default_prob / default_prob.max() * 0.3).astype(int)
    
    # DataFrame
    X = np.column_stack([income, age, debt_ratio, credit_score, employment_years])
    feature_names = ['income', 'age', 'debt_ratio', 'credit_score', 'employment_years']
    
    print(f"   Записів: {len(X):,}")
    print(f"   Ознак: {len(feature_names)}")
    print(f"   Default rate: {y.mean():.1%}")
    
    # 2. Розбиття даних
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Навчання моделі
    print("\n🌳 Навчання моделі...")
    model = DecisionTreeClassifier(max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"   Accuracy: {accuracy:.1%}")
    
    # 4. Створення GPE Explainer
    print("\n🔧 Створення GPE Explainer...")
    explainer = GPEExplainer(
        model=model,
        feature_names=feature_names,
        X_train=X_train,
        min_precision=0.95
    )
    
    # 5. Пояснення для кількох клієнтів
    print("\n" + "="*60)
    print("  ПРИКЛАДИ ПОЯСНЕНЬ")
    print("="*60)
    
    # Знайти приклади дефолтів і не-дефолтів
    default_indices = np.where(y_test == 1)[0][:3]
    good_indices = np.where(y_test == 0)[0][:2]
    
    print("\n🔴 КЛІЄНТИ З ДЕФОЛТОМ:")
    for i, idx in enumerate(default_indices):
        x = X_test[idx]
        exp = explainer.explain(x, X_train)
        
        print(f"\n   Клієнт {i+1}:")
        print(f"   ├── Income: ${x[0]:,.0f}")
        print(f"   ├── Age: {x[1]:.0f}")
        print(f"   ├── Debt ratio: {x[2]:.1%}")
        print(f"   ├── Credit score: {x[3]:.0f}")
        print(f"   └── Employment: {x[4]:.1f} років")
        print(f"\n   📋 GPE Пояснення:")
        print(f"   └── {exp.rule}")
        print(f"       Precision: {exp.precision:.1%}, Coverage: {exp.coverage:.1%}")
        
        # Природна мова
        print(f"\n   💬 Природна мова:")
        print(f"       {exp.to_natural_language()}")
    
    print("\n🟢 КЛІЄНТИ БЕЗ ДЕФОЛТУ:")
    for i, idx in enumerate(good_indices):
        x = X_test[idx]
        exp = explainer.explain(x, X_train)
        
        print(f"\n   Клієнт {i+1}:")
        print(f"   ├── Income: ${x[0]:,.0f}")
        print(f"   ├── Credit score: {x[3]:.0f}")
        print(f"\n   📋 GPE Пояснення:")
        print(f"   └── {exp.rule}")
        print(f"       Precision: {exp.precision:.1%}")
    
    # 6. Порівняння з GPE-IT
    print("\n" + "="*60)
    print("  ПОРІВНЯННЯ GPE-CORE vs GPE-IT")
    print("="*60)
    
    gpe_it = GPEInformationTheoretic(
        model=model,
        feature_names=feature_names,
        X_train=X_train
    )
    
    x_example = X_test[default_indices[0]]
    
    exp_core = explainer.explain(x_example, X_train)
    exp_it = gpe_it.explain(x_example, X_train)
    
    print(f"\n   GPE-Core: {exp_core.rule}")
    print(f"   GPE-IT:   {exp_it.rule}")
    
    # 7. Бенчмарк швидкості
    print("\n" + "="*60)
    print("  БЕНЧМАРК ШВИДКОСТІ")
    print("="*60)
    
    import time
    
    n_test = 100
    
    # GPE
    start = time.time()
    for x in X_test[:n_test]:
        _ = explainer.explain(x, X_train)
    gpe_time = (time.time() - start) / n_test * 1000
    
    print(f"\n   GPE-Core: {gpe_time:.2f} ms per explanation")
    
    # GPE-IT
    start = time.time()
    for x in X_test[:n_test]:
        _ = gpe_it.explain(x, X_train)
    gpe_it_time = (time.time() - start) / n_test * 1000
    
    print(f"   GPE-IT:   {gpe_it_time:.2f} ms per explanation")
    
    # LIME (якщо встановлено)
    try:
        import lime.lime_tabular
        
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train[:1000],
            feature_names=feature_names,
            mode='classification'
        )
        
        start = time.time()
        for x in X_test[:30]:  # LIME повільний
            _ = lime_explainer.explain_instance(x, model.predict_proba, num_features=5)
        lime_time = (time.time() - start) / 30 * 1000
        
        print(f"   LIME:     {lime_time:.2f} ms per explanation")
        print(f"\n   📈 GPE швидший за LIME в {lime_time/gpe_time:.0f}x разів!")
    except ImportError:
        print("   LIME не встановлено")
    
    print("\n" + "="*60)
    print("  ДЕМОНСТРАЦІЯ ЗАВЕРШЕНА!")
    print("="*60)


if __name__ == '__main__':
    main()

