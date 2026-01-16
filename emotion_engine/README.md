
# Emotion Engine

A production-ready, modular Python engine for continuous, dynamic, and interpretable emotion modeling, inference, reasoning, and visualization. Runs fully offline in the Windows terminal. Supports both classical ML and transformer-based (DistilBERT) emotion inference.

---

## Features
- Hybrid VAD (Valence-Arousal-Dominance) + Discrete emotion modeling
- Emotion dynamics: blending, decay, memory, transitions
- Classical ML and transformer (DistilBERT) inference
- Reasoning and explainability (SHAP, feature importance)
- File-based visualizations (PNG/JPG): confusion matrix, metrics, VAD plots
- Logging, reproducibility, and offline operation
- Modular, extensible architecture

---

## Installation

1. Clone the repository:
	```bash
	git clone <repo-url>
	cd emotion_engine
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
	- For DistilBERT support: `transformers`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`

---

## Usage

Run the engine from the terminal:
```bash
python main.py
```

You will be prompted to select the inference engine:
- Option 1: Classical ML (Logistic Regression, SVM, etc.)
- Option 2: DistilBERT (transformer-based)

Results (metrics and visualizations) are saved to the `output/` folder.

To evaluate and compare both models:
```bash
python emotion_engine/evaluate_all.py
```

---

## Project Structure

- `core/` — State, memory, config management
- `inference/` — ML pipeline, DistilBERT, explainability
- `dynamics/` — Emotion transitions and blending
- `reasoning/` — Reasoning engine, explainability
- `visualization/` — Plots: confusion matrix, VAD, metrics
- `utils/` — Logging, file I/O
- `tests/` — Unit tests

---

## Dataset Size Requirements

| Level                            | Dataset Size (Text Samples) | What You Can Achieve              |
| -------------------------------- | --------------------------- | --------------------------------- |
| Minimum (Prototype)              | 5k – 10k                    | Works, demo-ready, basic accuracy |
| Good (Portfolio / Interview)     | 20k – 40k                   | Stable, reliable, explainable     |
| Strong (Research / Product)      | 80k – 150k                  | Robust, generalizable             |
| Very Advanced (Near-SOTA)        | 300k+                       | Industry-grade (optional)         |

See below for practical guidance on dataset selection and augmentation.

---

## Classical ML and DistilBERT Support

This project supports emotion inference using both classical ML and Hugging Face transformer models (DistilBERT).

### Classical ML
- Uses TF-IDF vectorization and models like Logistic Regression, SVM.
- Fast, interpretable, works well with moderate data.
- See `inference/text_inference.py` for implementation.

### DistilBERT (Transformers)
- Uses Hugging Face `transformers` and `torch` for deep learning-based emotion classification.
- Supports fine-tuning on custom emotion datasets.
- See `run_transformers_demo.py` for usage example.
- To fine-tune: see comments in `inference/model_training.py`.

---

## Evaluation and Visualization

After running the engine or evaluation script, you will get:
- Precision, recall, F1-score, accuracy for each model
- Confusion matrix (PNG/JPG)
- Bar charts comparing metrics for classical ML and DistilBERT
- VAD plots and dataset distribution visualizations

All outputs are saved in the `output/` folder for review.

---

## Extensibility and Offline Operation

- Fully modular: add new models, reasoning engines, or visualization types easily
- Runs offline: no internet required after initial setup
- Supports custom datasets and emotion taxonomies
- Easy to integrate with other Python projects

---

## Troubleshooting & FAQ

**Q: DistilBERT metrics are low or show warnings?**
A: Ensure your emotion labels match the pre-trained model’s expected classes, or fine-tune DistilBERT on your custom dataset.

**Q: How do I add new emotions or change the taxonomy?**
A: Update the label set in your dataset and retrain both models. For VAD mapping, update the config in `core/`.

**Q: Where are the results saved?**
A: All metrics and visualizations are saved in the `output/` folder.

**Q: Can I run everything offline?**
A: Yes, after installing dependencies, all inference and visualization is fully offline.

---

## License & Credits

MIT License. Built using Python, scikit-learn, Hugging Face Transformers, matplotlib, seaborn, pandas, numpy.

---

## Contact

For questions, suggestions, or contributions, open an issue or pull request.

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

## DistilBERT and Hugging Face Transformers Support

This project supports emotion inference using DistilBERT and other Hugging Face models. To use DistilBERT-based emotion classification:

- Install requirements: `pip install -r requirements.txt`
- Use the `DistilBERTEmotionInference` class in `inference/text_inference.py` for transformer-based emotion inference.
- See `run_transformers_demo.py` for a usage example.

You can fine-tune DistilBERT for your own emotion dataset using Hugging Face Trainer (see comments in `model_training.py`).
