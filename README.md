# Redact — Interactive Machine Unlearning Lab

> *Redact* — to selectively remove sensitive information from a record, leaving the rest intact.

Redact is an interactive proof-of-concept for the **UNSIR** (Unlearning by Selective Impair and Repair) framework introduced in:

> Tarun, A. K., Chundawat, V. S., Mandal, M., & Kankanhalli, M.
> *Fast Yet Effective Machine Unlearning.* arXiv:2111.08947v5, May 2023.

It implements zero-glance class-level machine unlearning on a ResNet-18 trained on CIFAR-10, and exposes the full pipeline through a Flask-backed web dashboard. Pick the classes you want a trained model to forget; watch it forget them, in seconds, without ever revisiting a single forget-class sample.

---

## What it demonstrates

1. **Phase 1 — Error-Maximizing Noise Generation.** A noise tensor `N` (shape of input) is optimized so that the *frozen* network classifies it confidently as the forget class. `N` becomes an "anti-pattern" of that class — a learned input that maximally fires the to-be-forgotten neurons.
2. **Phase 2 — Impair Step.** One epoch on `Dr_sub ∪ N` at high LR (0.02). The anti-pattern pushes the model away from the forget class while a small retain subset stabilises shared features.
3. **Phase 3 — Repair Step.** One epoch on `Dr_sub` only at low LR (0.01). Restores accuracy on retained classes without ever touching the forget set.

The whole impair+repair runs in seconds and never sees a single forget-class sample.

## Interactive dashboard

The Flask app at `http://localhost:5000` lets you:

- view per-class accuracy on the test set for the current model,
- select one or more classes to forget,
- run UNSIR live, watching the noise generation, impair, and repair phases stream their metrics,
- compare *before* and *after* per-class accuracy (forget accuracy → ~0%, retain accuracy preserved),
- reset to the original baseline weights and try a different forget set.

## Run it

```bash
pip install -r requirements.txt

# train baseline ResNet-18 on CIFAR-10 (~5 min on GPU)
python -m redact.train

# launch dashboard
python -m redact.app
```

Then open <http://localhost:5000>.

## Layout

```
redact/
  model.py     CIFAR-adapted ResNet-18
  data.py      CIFAR-10 loaders, retain/forget splits
  unsir.py     noise generation + impair + repair
  train.py     baseline trainer
  evaluate.py  per-class accuracy + sample predictions
  app.py       Flask dashboard
templates/     dashboard HTML
static/        dashboard CSS/JS
tests/         smoke tests for the unlearning pipeline
checkpoints/   saved baseline weights
```

## Metrics reported

- `ADf` — accuracy on forget classes (target ≈ 0%)
- `ADr` — accuracy on retain classes (target close to original)
- per-phase wall-clock time (paper claims < 4 s end-to-end on ResNet18+CIFAR-10)

## Why "Redact"?

When a privacy lawyer redacts a document, they black out specific sentences while leaving the rest perfectly readable. UNSIR is the same idea applied to a neural network: kill class-`c` knowledge, keep everything else intact. The name maps to the paper's GDPR / right-to-erasure motivation and to the surgical, *selective* nature of the algorithm itself.
