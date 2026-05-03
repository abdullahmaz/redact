"""Flask dashboard for Redact.

Endpoints:
    GET  /                — dashboard UI
    GET  /api/status      — device, model, baseline accs, current accs, history
    GET  /api/classes     — CIFAR-10 class names
    POST /api/unlearn     — body: { forget_classes: [int, ...], steps?: int, ... }
    POST /api/reset       — restore baseline weights
    GET  /api/sample/<i>  — PNG of test image i (un-normalised)
    GET  /api/predict/<i> — softmax probabilities for test image i
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, render_template, Response

from .data import CIFAR10_CLASSES, retain_subset
from .evaluate import per_class_accuracy
from .state import init_state, tensor_to_png_bytes
from .unsir import UnsirConfig, run_unsir


def create_app(checkpoint_path: str = "checkpoints/baseline.pt") -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent.parent / "templates"),
        static_folder=str(Path(__file__).parent.parent / "static"),
    )
    state = init_state(checkpoint_path=checkpoint_path)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/classes")
    def classes():
        return jsonify({"classes": CIFAR10_CLASSES})

    @app.route("/api/status")
    def status():
        return jsonify({
            "device": str(state.device),
            "checkpoint": state.checkpoint_path,
            "classes": CIFAR10_CLASSES,
            "baseline_per_class": state.baseline_per_class,
            "current_per_class": state.current_per_class,
            "history": state.history,
        })

    @app.route("/api/reset", methods=["POST"])
    def reset():
        with state.lock:
            state.reload_model()
            state.current_per_class = list(state.baseline_per_class)
            state.history = []
        return jsonify({"ok": True, "current_per_class": state.current_per_class})

    @app.route("/api/sample/<int:idx>")
    def sample(idx: int):
        if idx < 0 or idx >= len(state.test_set):
            return Response("out of range", status=404)
        img, _ = state.raw_test_set[idx]
        png = tensor_to_png_bytes(img)
        return Response(png, mimetype="image/png")

    @app.route("/api/predict/<int:idx>")
    def predict(idx: int):
        if idx < 0 or idx >= len(state.test_set):
            return Response("out of range", status=404)
        import torch
        img, lbl = state.test_set[idx]
        with torch.no_grad():
            logits = state.model(img.unsqueeze(0).to(state.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        return jsonify({
            "index": idx,
            "true": int(lbl),
            "probs": probs,
            "pred": int(max(range(10), key=lambda c: probs[c])),
        })

    @app.route("/api/unlearn", methods=["POST"])
    def unlearn():
        body = request.get_json(force=True) or {}
        forget = list(int(c) for c in body.get("forget_classes", []))
        if not forget:
            return jsonify({"error": "forget_classes required"}), 400
        if any(c < 0 or c >= 10 for c in forget):
            return jsonify({"error": "class index out of range"}), 400

        cfg = UnsirConfig(
            forget_classes=forget,
            noise_steps=int(body.get("noise_steps", 40)),
            noise_lr=float(body.get("noise_lr", 0.1)),
            noise_lambda=float(body.get("noise_lambda", 0.1)),
            noise_batch_size=int(body.get("noise_batch_size", 256)),
            impair_epochs=int(body.get("impair_epochs", 1)),
            impair_lr=float(body.get("impair_lr", 0.02)),
            repair_epochs=int(body.get("repair_epochs", 1)),
            repair_lr=float(body.get("repair_lr", 0.01)),
        )

        events: list[dict] = []

        def progress(name, payload):
            events.append({"event": name, **payload})

        with state.lock:
            t0 = time.perf_counter()
            retain_sub = retain_subset(state.train_set, forget,
                                       per_class=int(body.get("per_class", 1000)))
            timings = run_unsir(state.model, retain_sub, cfg, progress=progress)
            new_accs = per_class_accuracy(state.model, state.test_loader, 10, state.device)
            state.current_per_class = new_accs
            adf = sum(new_accs[c] for c in forget) / len(forget)
            retain_classes = [c for c in range(10) if c not in set(forget)]
            adr = sum(new_accs[c] for c in retain_classes) / len(retain_classes)
            entry = {
                "forget_classes": forget,
                "timings": timings,
                "ADf": adf,
                "ADr": adr,
                "per_class": new_accs,
                "wall_time": time.perf_counter() - t0,
            }
            state.history.append(entry)

        # Trim verbose per-step events for response payload size.
        slim_events = [e for e in events
                       if e["event"] not in {"noise_step", "impair_batch", "repair_batch"}]
        return jsonify({
            "ok": True,
            "result": entry,
            "events": slim_events,
            "step_counts": {
                "noise_steps": sum(1 for e in events if e["event"] == "noise_step"),
                "impair_batches": sum(1 for e in events if e["event"] == "impair_batch"),
                "repair_batches": sum(1 for e in events if e["event"] == "repair_batch"),
            },
        })

    return app


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    create_app(args.checkpoint).run(host=args.host, port=args.port, debug=False)
