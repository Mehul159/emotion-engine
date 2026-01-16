# Emotion Engine

A production-ready, modular Python engine for continuous, dynamic, and interpretable emotion modeling, inference, and visualization. Runs fully offline in the Windows terminal.

## Features
- Hybrid VAD + Discrete emotion model
- Emotion dynamics, blending, decay, memory
- Offline ML-based text inference
- Reasoning and explainability
- File-based visualizations (PNG/JPG)
- Logging and reproducibility

## Usage
```bash
python main.py
```

## Structure
- core/: State, memory, config
- inference/: ML pipeline, explainability
- dynamics/: Emotion transitions
- reasoning/: Reasoning engine
- visualization/: Plots
- utils/: Logging, file I/O
- tests/: Unit tests

## Dataset Size Requirements for Emotion Engine

The table below summarizes recommended dataset sizes for different maturity levels of the Emotion Engine, with practical reasoning for each stage.

| Level                            | Dataset Size (Text Samples) | What You Can Achieve              |
| -------------------------------- | --------------------------- | --------------------------------- |
| Minimum (Prototype)              | 5k – 10k                    | Works, demo-ready, basic accuracy |
| Good (Portfolio / Interview)     | 20k – 40k                   | Stable, reliable, explainable     |
| Strong (Research / Product)      | 80k – 150k                  | Robust, generalizable             |
| Very Advanced (Near-SOTA)        | 300k+                       | Industry-grade (optional)         |

### Level Explanations

**Minimum (Prototype) — 5k–10k samples**
- **Why sufficient:** Enough for basic ML models (Logistic Regression, SVM) to learn major emotion classes and demonstrate the pipeline.
- **Suitable models:** Classical ML (TF-IDF + Logistic/SVM), simple neural nets.
- **Limitations:** Lower accuracy, less stable on rare emotions, limited generalization.
- **Trade-offs:** Fast iteration, low compute, but not robust to edge cases.
- **Use case:** Demos, hackathons, proof-of-concept.

**Good (Portfolio / Interview) — 20k–40k samples**
- **Why sufficient:** Enables more reliable, explainable models with better class coverage and confidence calibration.
- **Suitable models:** Classical ML, shallow neural nets, basic LSTM.
- **Limitations:** May still struggle with rare/ambiguous emotions, but stable for most use cases.
- **Trade-offs:** Good balance of accuracy and complexity, manageable training time.
- **Use case:** Portfolio projects, interviews, small-scale research.

**Strong (Research / Product) — 80k–150k samples**
- **Why sufficient:** Supports robust, generalizable models, including deeper neural networks and hybrid approaches.
- **Suitable models:** Deep learning (LSTM, CNN, small transformers), advanced ML ensembles.
- **Limitations:** Diminishing returns above this range for most applications; rare classes may still need augmentation.
- **Trade-offs:** High accuracy, robust to noise, but requires more compute and careful validation.
- **Use case:** Academic research, production pilots, commercial prototypes.

**Very Advanced (Near-SOTA) — 300k+ samples**
- **Why sufficient:** Approaches industry-grade performance, supports large transformer models and fine-tuning.
- **Suitable models:** Large transformers (BERT, RoBERTa), custom architectures.
- **Limitations:** Significant compute/resources required; further data yields diminishing returns unless targeting rare/complex emotions.
- **Trade-offs:** Highest accuracy and generalization, but with increased cost and complexity.
- **Use case:** Large-scale products, SOTA research, industry deployment (optional for most projects).

---

### Additional Guidance

- **Hybrid emotion engines (ML + rules + dynamics) require LESS data:** By combining ML with rule-based logic and emotion dynamics, the system can achieve high interpretability and stability even with moderate data sizes. Rules and dynamics compensate for data gaps and improve generalization.
- **Perfect class balance is NOT mandatory:** Real-world emotion data is naturally imbalanced. Modern ML models (with class weighting, calibration, and augmentation) handle imbalance well. Over-balancing can reduce realism.
- **VAD-based modeling reduces data dependency:** Mapping discrete emotions to continuous VAD space allows the engine to interpolate and blend emotions, making it less sensitive to data sparsity and class imbalance.
- **Synthetic and augmented data:** Generating synthetic samples or augmenting existing data (paraphrasing, back-translation) is a safe and effective way to expand datasets, especially for rare emotions or edge cases.

**Summary:**  
You do not need massive datasets to build a practical, production-ready emotion engine. Start with 5k–10k samples for prototypes, 20k–40k for reliable models, and 80k+ for robust research or product use. Hybrid approaches, VAD modeling, and data augmentation further reduce data requirements and improve system stability.
