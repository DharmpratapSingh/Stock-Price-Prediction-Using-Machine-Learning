"""
Inference CLI for next-day return prediction.

Feature parity with training is structural, not a convention that has to be
remembered: train.py saves one artifact holding the fitted pipeline, the exact
ordered feature list and the config that built it, and this module rebuilds
features by calling the same ``build_dataset`` with that same config, then
selects the artifact's feature list by name. If the two ever diverge, the
mismatch is raised here rather than silently producing a wrong number.

The output is a forecast *log return* and its direction, never a price level
presented as a precise target. A point forecast of tomorrow's close would look
authoritative while carrying essentially no information beyond today's price.

Usage:
    python predict.py --model models/NVDA_linear.joblib
    python predict.py --model models/NVDA_linear.joblib --symbol AAPL
    python predict.py --model models/NVDA_linear.joblib --batch --symbols SPY AAPL MSFT
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_loader import load_stock_data
from src.feature_engineering import build_dataset
from src.utils import load_artifact, setup_logging

logger = logging.getLogger(__name__)


def build_inference_features(
    symbol: str,
    artifact: Dict,
    allow_download: bool = False
) -> pd.DataFrame:
    """
    Rebuild the model matrix for a ticker exactly as training built it.

    Args:
        symbol: Ticker symbol
        artifact: Loaded training artifact
        allow_download: Permit a network fetch when no snapshot exists

    Returns:
        DataFrame of engineered features, restricted and ordered to match the
        artifact's feature list, with the raw OHLCV columns still attached

    Raises:
        ValueError: if any training feature is missing at inference time
    """
    config = artifact["config"]
    data_cfg = config["data"]

    raw = load_stock_data(
        symbol=symbol,
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        snapshot_dir=data_cfg.get("snapshot_dir", "data/raw"),
        allow_download=allow_download or data_cfg.get("allow_download", False),
    )

    dataset, _ = build_dataset(
        raw,
        config["features"],
        horizon=artifact.get("horizon", 1),
        stationary_only=config["features"].get("stationary_only", True),
    )

    expected = list(artifact["feature_columns"])
    missing = [c for c in expected if c not in dataset.columns]
    if missing:
        raise ValueError(
            f"Feature parity check failed for {symbol}: the artifact was fitted on "
            f"{len(expected)} features but {len(missing)} are missing from the "
            f"inference matrix: {missing[:10]}. Retrain with the current config."
        )

    return dataset


def predict(
    model_path: str,
    symbol: str = None,
    allow_download: bool = False,
    n_recent: int = 1
) -> pd.DataFrame:
    """
    Forecast the next-day log return for the most recent available bar(s).

    Args:
        model_path: Path to a train.py artifact
        symbol: Ticker to predict (defaults to the artifact's training ticker)
        allow_download: Permit a network fetch
        n_recent: How many trailing bars to score

    Returns:
        DataFrame with one row per scored bar
    """
    artifact = load_artifact(model_path)
    symbol = (symbol or artifact["symbol"]).upper()

    logger.info(
        "Artifact: %s family, trained on %s through %s (%d rows, %d features)",
        artifact.get("family"), artifact.get("symbol"),
        artifact.get("trained_through"), artifact.get("n_train_rows", 0),
        len(artifact["feature_columns"]),
    )

    dataset = build_inference_features(symbol, artifact, allow_download)
    features = artifact["feature_columns"]

    recent = dataset.iloc[-n_recent:]
    X = recent[features]

    predicted_log_return = np.asarray(artifact["pipeline"].predict(X), dtype=float)

    direction_pipeline = artifact.get("direction_pipeline")
    if direction_pipeline is not None:
        up_probability = direction_pipeline.predict_proba(X)[:, 1]
    else:
        up_probability = np.full(len(recent), np.nan)

    spot = recent["Close"].to_numpy()

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "as_of": recent.index.date,
            "spot_close": spot,
            "predicted_log_return": predicted_log_return,
            "predicted_pct_move": np.expm1(predicted_log_return) * 100,
            "implied_next_close": spot * np.exp(predicted_log_return),
            "direction": np.where(predicted_log_return > 0, "up", "down"),
            "p_up": up_probability,
        }
    )

    # The realised outcome is known for every bar except the last one, so the
    # CLI can be checked against reality rather than taken on trust.
    out["realised_log_return"] = recent["target_logret"].to_numpy()
    return out


def batch_predict(
    model_path: str,
    symbols: List[str],
    output_file: str = None,
    allow_download: bool = False
) -> pd.DataFrame:
    """
    Score several tickers with one artifact.

    Args:
        model_path: Path to a train.py artifact
        symbols: Tickers to score
        output_file: Optional CSV destination
        allow_download: Permit a network fetch

    Returns:
        Concatenated predictions
    """
    frames = []
    for symbol in symbols:
        try:
            frames.append(predict(model_path, symbol, allow_download))
        except Exception as exc:
            logger.error("Skipping %s: %s", symbol, exc)

    if not frames:
        raise RuntimeError("No symbol could be scored")

    result = pd.concat(frames, ignore_index=True)
    if output_file:
        result.to_csv(output_file, index=False)
        logger.info("Wrote %s", output_file)
    return result


def print_predictions(frame: pd.DataFrame) -> None:
    """Print a forecast table with the honest caveat attached."""
    print("\n" + "=" * 78)
    print("NEXT-DAY RETURN FORECAST")
    print("=" * 78)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("-" * 78)
    print(
        "Walk-forward testing puts this model's out-of-sample return R2 at roughly\n"
        "zero and its directional accuracy within sampling error of a coin flip.\n"
        "Treat these as an illustration of the pipeline, not a trading signal."
    )
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast the next-day log return from a trained artifact"
    )
    parser.add_argument("--model", help="Path to a train.py artifact (.joblib)")
    parser.add_argument("--symbol", help="Ticker to score")
    parser.add_argument("--symbols", nargs="+", help="Tickers for batch mode")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--recent", type=int, default=1,
                        help="Number of trailing bars to score")
    parser.add_argument("--download", action="store_true",
                        help="Allow fetching data not present as a snapshot")
    parser.add_argument("--output", help="CSV output path for batch mode")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    setup_logging(args.log_level)

    model_path = args.model
    if not model_path:
        candidates = sorted(Path("models").glob("*.joblib"))
        if not candidates:
            parser.error("No --model given and no artifact found in models/. Run train.py first.")
        model_path = str(candidates[-1])
        logger.info("No --model given, using %s", model_path)

    if args.batch:
        symbols = args.symbols or [args.symbol]
        if not symbols or symbols == [None]:
            parser.error("--batch requires --symbols")
        frame = batch_predict(model_path, symbols, args.output, args.download)
    else:
        frame = predict(model_path, args.symbol, args.download, args.recent)

    print_predictions(frame)


if __name__ == "__main__":
    main()
