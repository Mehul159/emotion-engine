"""
Explainability (SHAP/Textual)
"""

import shap
import numpy as np

def explain_prediction(model, vectorizer, text):
    X = vectorizer.transform([text])
    explainer = shap.Explainer(model.estimators_[0], vectorizer)
    shap_values = explainer(X)
    # Get top features
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_indices = np.argsort(-np.abs(shap_values.values[0]))[:5]
    top_features = feature_names[top_indices]
    top_scores = shap_values.values[0][top_indices]
    explanation = "Top features: " + ", ".join(f"{f} ({s:.2f})" for f, s in zip(top_features, top_scores))
    return explanation
