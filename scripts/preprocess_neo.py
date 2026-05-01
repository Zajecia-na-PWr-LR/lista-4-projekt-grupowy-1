from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "neo_v2.csv"
OUTPUT_CSV = ROOT / "data" / "neo_preprocessed.csv"

DROP_NAME_BODY_SENTRY = ["name", "orbiting_body", "sentry_object"]
DROP_REDUNDANT_DIAMETER = "est_diameter_max"


def preprocess_neo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in DROP_NAME_BODY_SENTRY if c not in out.columns]
    if missing:
        raise ValueError(f"Expected columns missing from input: {missing}")
    if DROP_REDUNDANT_DIAMETER not in out.columns:
        raise ValueError(f"Expected column {DROP_REDUNDANT_DIAMETER!r} not in input.")
    out = out.drop(columns=DROP_NAME_BODY_SENTRY + [DROP_REDUNDANT_DIAMETER])
    max_diameters = out.groupby("id")["est_diameter_min"].max().nlargest(2).index
    out = out[~out["id"].isin(max_diameters)]
    return out


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    processed = preprocess_neo(df)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUTPUT_CSV, index=False)
    print(
        f"Wrote {len(processed)} rows, {len(processed.columns)} columns -> {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
