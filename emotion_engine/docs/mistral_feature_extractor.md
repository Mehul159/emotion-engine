# Mistral LLM Feature Extraction Usage

## Purpose
The `MistralFeatureExtractor` provides tokenization and (optionally) model-based features for assistive, explainable emotion signal extraction in the Emotion Engine. It is never used for end-to-end emotion prediction or direct emotion decisions.

## Usage Example
```
from ml_assist.mistral_feature_extractor import MistralFeatureExtractor
extractor = MistralFeatureExtractor()
token_count = extractor.get_token_count("Your input text here.")
```

- Use `token_count` as an additional, explainable feature in your normalization logic (e.g., longer input = higher arousal/confidence).
- The extractor can be extended to provide more advanced features if needed.

## Compliance
- All outputs are bounded, explainable, and policy-checked.
- No black-box emotion prediction is performed.

## Integration
- Used in `FeatureNormalizer` and can be imported anywhere in the pipeline.
- Update or extend as new Mistral models or features become available.
