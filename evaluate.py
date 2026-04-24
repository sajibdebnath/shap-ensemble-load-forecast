"""
Inference and SHAP Interpretability Script.

Run post-training evaluation and SHAP attribution on the test set.

Usage
-----
  python evaluate.py --ercot_path data/raw/ercot_load_2018_2025.csv \
                     --asos_paths data/raw/asos_BKS.csv data/raw/asos_JDD.csv data/raw/asos_TME.csv \
                     --checkpoint_dir results/checkpoints \
                     --run_name pi_ensemble
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.cnn_branch import build_cnn_branch
from src.models.transformer_branch import build_transformer_branch
from src.ensemble.weighted_ensemble import WeightedEnsemble
from src.explainability.shap_explainer import EnsembleSHAPExplainer, hampel_filter
from src.preprocessing.data_pipeline import (
    load_ercot,
    load_and_merge_asos_stations,
    build_features,
    temporal_split,
    make_sequences,
    apply_scaler,
)
from src.evaluation.metrics import evaluate, evaluate_extreme_events, ablation_table
from src.visualization.plots import (
    plot_parabolic_envelope,
    plot_global_shap_importance,
    plot_regime_shap_comparison,
    plot_test_timeseries,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PI ensemble and run SHAP attribution."
    )
    parser.add_argument("--ercot_path", type=str, required=True)
    parser.add_argument("--asos_paths", type=str, nargs="+", required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--run_name", type=str, default="pi_ensemble")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--shap_background", type=int, default=500,
                        help="Number of background samples for SHAP (default: 500).")
    parser.add_argument("--shap_n_bootstraps", type=int, default=20,
                        help="Bootstrap resamples for Kendall τ stability (default: 20).")
    parser.add_argument("--skip_shap", action="store_true",
                        help="Skip SHAP computation (faster evaluation-only run).")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)

    # ── 1. Load and preprocess data ───────────────────────────────────────────
    print("[1/6] Loading and preprocessing data …")
    ercot_df = load_ercot(args.ercot_path)
    asos_df = load_and_merge_asos_stations(args.asos_paths)
    feature_df = build_features(ercot_df, asos_df)

    train_df, val_df, test_df = temporal_split(feature_df)

    X_train, y_train, T_train = make_sequences(train_df, seq_len=args.seq_len)
    X_val,   y_val,   T_val   = make_sequences(val_df,   seq_len=args.seq_len)
    X_test,  y_test,  T_test  = make_sequences(test_df,  seq_len=args.seq_len)

    with open(ckpt_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    _, _, F = X_train.shape

    # ── 2. Restore trained models ─────────────────────────────────────────────
    print("[2/6] Restoring branch model weights …")
    cnn_model = build_cnn_branch(seq_len=args.seq_len, n_features=F)
    transformer_model = build_transformer_branch(seq_len=args.seq_len, n_features=F)

    cnn_model.load_weights(str(ckpt_dir / f"{args.run_name}_cnn_best.weights.h5"))
    transformer_model.load_weights(str(ckpt_dir / f"{args.run_name}_transformer_best.weights.h5"))

    # ── 3. Generate predictions ───────────────────────────────────────────────
    print("[3/6] Generating predictions …")

    def predict_branch(model, X, batch_size=256):
        preds = []
        for i in range(0, len(X), batch_size):
            _, y_hat = model(X[i: i + batch_size], training=False)
            preds.append(y_hat.numpy().squeeze())
        return np.concatenate(preds)

    y_pred_cnn = predict_branch(cnn_model, X_test)
    y_pred_transformer = predict_branch(transformer_model, X_test)

    # Fit ensemble weights on validation set
    y_val_cnn = predict_branch(cnn_model, X_val)
    y_val_transformer = predict_branch(transformer_model, X_val)
    ensemble = WeightedEnsemble.from_validation(y_val, y_val_cnn, y_val_transformer)
    y_pred_ensemble = ensemble.predict(y_pred_cnn, y_pred_transformer)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/6] Evaluation results …")
    print("\n── Overall performance (Table II) ──")
    for label, y_pred in [
        ("CNN branch (PI)", y_pred_cnn),
        ("Transformer branch (PI)", y_pred_transformer),
        ("PI Ensemble", y_pred_ensemble),
    ]:
        evaluate(y_test, y_pred, label=label)

    print("\n── Extreme-event performance (Table III) ──")
    extreme_df = evaluate_extreme_events(
        y_test,
        {
            "CNN branch (PI)": y_pred_cnn,
            "Transformer branch (PI)": y_pred_transformer,
            "PI Ensemble": y_pred_ensemble,
        }
    )
    print(extreme_df.to_string())
    extreme_df.to_csv("results/tables/table3_extreme_event_performance.csv")

    # ── 5. Generate visualisations ────────────────────────────────────────────
    print("\n[5/6] Generating figures …")
    extreme_mask = hampel_filter(y_test)

    # Fig. 3(a) – parabolic envelope
    plot_parabolic_envelope(T_train, y_train)

    # Fig. 5 – test-year time series
    test_timestamps = test_df.index[args.seq_len:]
    plot_test_timeseries(test_timestamps, y_test, y_pred_ensemble, extreme_mask)

    # ── 6. SHAP attribution ───────────────────────────────────────────────────
    if not args.skip_shap:
        print("\n[6/6] Computing SHAP attributions …")
        explainer = EnsembleSHAPExplainer(
            cnn_model=cnn_model,
            transformer_model=transformer_model,
            w_cnn=ensemble.w_cnn_,
            w_transformer=ensemble.w_transformer_,
            background_size=args.shap_background,
        )
        explainer.fit(X_train, y_train)

        phi = explainer.compute_shap_values(X_test)

        global_imp = explainer.global_importance(phi)
        normal_imp, extreme_imp = explainer.regime_comparison(phi, extreme_mask)

        print("\n── Global SHAP importance (Fig. 7) ──")
        for feat, val in global_imp.items():
            print(f"  {feat:20s}: {val:,.1f} MW")

        print("\n── Regime shift: wind_speed ──")
        print(f"  Normal    : {normal_imp.get('wind_speed', 0):,.1f} MW")
        print(f"  Extreme   : {extreme_imp.get('wind_speed', 0):,.1f} MW")

        plot_global_shap_importance(global_imp)
        plot_regime_shap_comparison(normal_imp, extreme_imp)

        # Rank stability (Kendall τ)
        tau = explainer.bootstrap_rank_stability(
            X_train, y_train, X_test,
            n_bootstraps=args.shap_n_bootstraps,
        )
        print(f"\n  Kendall τ rank-stability: {tau:.3f} (paper reports 0.91)")

        # Save SHAP values
        np.save("results/shap_values.npy", phi)
        print("  SHAP values saved → results/shap_values.npy")
    else:
        print("[6/6] SHAP computation skipped (--skip_shap).")

    print("\n✓ Evaluation complete. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
