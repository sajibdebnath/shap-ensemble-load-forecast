"""
Main Training Script — Physics-Informed CNN–Transformer Ensemble.

Reproduces the training pipeline described in:
  Abubakkar et al., "Interpretable Physics-Informed Load Forecasting for
  U.S. Grid Resilience", IEEE, 2026.

Usage
-----
  # Train with default settings (ERCOT 2018–2025)
  python train.py --ercot_path data/raw/ercot_load_2018_2025.csv \
                  --asos_paths data/raw/asos_BKS.csv data/raw/asos_JDD.csv data/raw/asos_TME.csv

  # Custom lambda weights
  python train.py ... --lambda1 0.1 --lambda2 0.05

  # Skip physics constraints (ablation)
  python train.py ... --lambda1 0.0 --lambda2 0.0 --run_name "no_physics"
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.cnn_branch import build_cnn_branch
from src.models.transformer_branch import build_transformer_branch
from src.losses.physics_loss import physics_informed_loss
from src.ensemble.weighted_ensemble import WeightedEnsemble
from src.preprocessing.data_pipeline import (
    load_ercot,
    load_and_merge_asos_stations,
    build_features,
    temporal_split,
    make_sequences,
    fit_scaler,
    apply_scaler,
)
from src.evaluation.metrics import evaluate, evaluate_extreme_events

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# ── Custom training loop (physics-informed loss) ──────────────────────────────

class PhysicsInformedTrainer:
    """
    Custom training loop supporting the physics-informed composite loss.

    Using a custom loop allows the temperature tensor to be passed alongside
    y_true, which is required by the parabolic constraint term.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        lambda1: float = 0.1,
        lambda2: float = 0.05,
        delta_max: float = 4_800.0,
        sigma_mw: float = 3_000.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta_max = delta_max
        self.sigma_mw = sigma_mw

    @tf.function
    def train_step(
        self,
        X_batch: tf.Tensor,
        y_batch: tf.Tensor,
        temp_batch: tf.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            _, y_pred = self.model(X_batch, training=True)
            y_pred = tf.squeeze(y_pred)

            loss = physics_informed_loss(
                y_true=y_batch,
                y_pred=y_pred,
                temperature=temp_batch,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                delta_max=self.delta_max,
                sigma_mw=self.sigma_mw,
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return loss

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        preds = []
        for i in range(0, len(X), batch_size):
            _, y_hat = self.model(X[i: i + batch_size], training=False)
            preds.append(y_hat.numpy().squeeze())
        return np.concatenate(preds)


def train_branch(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    T_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambda1: float,
    lambda2: float,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    run_name: str = "branch",
) -> Tuple[PhysicsInformedTrainer, list]:
    """
    Train a single branch model under the physics-informed composite loss.

    Returns
    -------
    trainer : PhysicsInformedTrainer
    history : list of float — validation MAE per epoch
    """
    from typing import Tuple  # local import for type annotation in function

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999
    )
    trainer = PhysicsInformedTrainer(
        model=model, optimizer=optimizer, lambda1=lambda1, lambda2=lambda2
    )

    n_batches = len(X_train) // batch_size
    best_val_mae = np.inf
    no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(X_train))
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]
        T_shuf = T_train[idx]

        epoch_loss = 0.0
        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            loss = trainer.train_step(
                tf.constant(X_shuf[sl], dtype=tf.float32),
                tf.constant(y_shuf[sl], dtype=tf.float32),
                tf.constant(T_shuf[sl], dtype=tf.float32),
            )
            epoch_loss += loss.numpy()

        # ── Validation ────────────────────────────────────────────────────────
        y_val_pred = trainer.predict(X_val)
        val_mae = float(np.mean(np.abs(y_val - y_val_pred)))
        history.append(val_mae)

        if epoch % 10 == 0:
            print(
                f"  [{run_name}] Epoch {epoch:3d}/{max_epochs}  "
                f"train_loss={epoch_loss / n_batches:.4f}  "
                f"val_MAE={val_mae:,.0f} MW"
            )

        # ── Early stopping ────────────────────────────────────────────────────
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve = 0
            model.save_weights(f"results/checkpoints/{run_name}_best.weights.h5")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{run_name}] Early stopping at epoch {epoch}.")
                break

    # Restore best weights
    model.load_weights(f"results/checkpoints/{run_name}_best.weights.h5")
    print(f"  [{run_name}] Best val MAE: {best_val_mae:,.0f} MW")
    return trainer, history


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train physics-informed CNN–Transformer ensemble for ERCOT load forecasting."
    )
    parser.add_argument("--ercot_path", type=str, required=True,
                        help="Path to ERCOT hourly load CSV.")
    parser.add_argument("--asos_paths", type=str, nargs="+", required=True,
                        help="Path(s) to ASOS station CSV(s) (BKS, JDD, TME).")
    parser.add_argument("--lambda1", type=float, default=0.1,
                        help="Parabolic constraint weight (default: 0.1).")
    parser.add_argument("--lambda2", type=float, default=0.05,
                        help="Ramp constraint weight (default: 0.05).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--run_name", type=str, default="pi_ensemble",
                        help="Identifier appended to checkpoint filenames.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # ── 1. Load & preprocess data ─────────────────────────────────────────────
    print("\n[Step 1] Loading data …")
    ercot_df = load_ercot(args.ercot_path)
    asos_df = load_and_merge_asos_stations(args.asos_paths)
    feature_df = build_features(ercot_df, asos_df)

    # ── 2. Temporal split ─────────────────────────────────────────────────────
    print("[Step 2] Splitting data …")
    train_df, val_df, test_df = temporal_split(feature_df)

    X_train, y_train, T_train = make_sequences(train_df, seq_len=args.seq_len)
    X_val,   y_val,   T_val   = make_sequences(val_df,   seq_len=args.seq_len)
    X_test,  y_test,  T_test  = make_sequences(test_df,  seq_len=args.seq_len)

    # ── 3. Normalise ──────────────────────────────────────────────────────────
    print("[Step 3] Normalising features …")
    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    # Save scaler for inference
    with open("results/checkpoints/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── 4. Build models ───────────────────────────────────────────────────────
    print("[Step 4] Building branch models …")
    _, _, F = X_train.shape
    cnn_model = build_cnn_branch(seq_len=args.seq_len, n_features=F)
    transformer_model = build_transformer_branch(seq_len=args.seq_len, n_features=F)

    # ── 5. Train CNN branch ───────────────────────────────────────────────────
    print(f"\n[Step 5] Training CNN branch  (λ₁={args.lambda1}, λ₂={args.lambda2}) …")
    cnn_trainer, _ = train_branch(
        model=cnn_model,
        X_train=X_train, y_train=y_train, T_train=T_train,
        X_val=X_val, y_val=y_val,
        lambda1=args.lambda1, lambda2=args.lambda2,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, lr=args.lr,
        run_name=f"{args.run_name}_cnn",
    )

    # ── 6. Train Transformer branch ───────────────────────────────────────────
    print(f"\n[Step 6] Training Transformer branch  (λ₁={args.lambda1}, λ₂={args.lambda2}) …")
    transformer_trainer, _ = train_branch(
        model=transformer_model,
        X_train=X_train, y_train=y_train, T_train=T_train,
        X_val=X_val, y_val=y_val,
        lambda1=args.lambda1, lambda2=args.lambda2,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, lr=args.lr,
        run_name=f"{args.run_name}_transformer",
    )

    # ── 7. Fit ensemble weights on validation set ─────────────────────────────
    print("\n[Step 7] Fitting ensemble weights on validation set …")
    y_val_cnn = cnn_trainer.predict(X_val)
    y_val_transformer = transformer_trainer.predict(X_val)
    ensemble = WeightedEnsemble.from_validation(y_val, y_val_cnn, y_val_transformer)
    print(f"  {ensemble}")

    # ── 8. Evaluate on test set ───────────────────────────────────────────────
    print("\n[Step 8] Evaluating on 2024–2025 test set …")
    y_test_cnn = cnn_trainer.predict(X_test)
    y_test_transformer = transformer_trainer.predict(X_test)
    y_test_ensemble = ensemble.predict(y_test_cnn, y_test_transformer)

    predictions = {
        "CNN branch (PI)": y_test_cnn,
        "Transformer branch (PI)": y_test_transformer,
        "PI Ensemble": y_test_ensemble,
    }

    print("\n── Overall performance (Table II) ──")
    for label, y_pred in predictions.items():
        evaluate(y_test, y_pred, label=label)

    print("\n── Extreme-event performance (Table III) ──")
    extreme_results = evaluate_extreme_events(y_test, predictions)
    print(extreme_results.to_string())

    # ── 9. Save predictions ───────────────────────────────────────────────────
    np.save("results/y_test.npy", y_test)
    np.save("results/y_pred_cnn.npy", y_test_cnn)
    np.save("results/y_pred_transformer.npy", y_test_transformer)
    np.save("results/y_pred_ensemble.npy", y_test_ensemble)
    print("\n✓ Predictions saved to results/")


if __name__ == "__main__":
    main()
