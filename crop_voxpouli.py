#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Drop first N (uid,dataset) pairs for a given dataset value.")
    p.add_argument("input_csv", help="Path to input CSV")
    p.add_argument("output_csv", help="Path to write the filtered CSV")
    p.add_argument("--dataset", default="facebook/voxpopuli", help="Dataset value to target (default: facebook/voxpopuli)")
    p.add_argument("--pairs", type=int, default=1000, help="How many (uid,dataset) pairs to drop (default: 1000)")
    p.add_argument("--uid-col", default="uid", help="Name of the UID column (default: uid)")
    p.add_argument("--dataset-col", default="dataset", help="Name of the dataset column (default: dataset)")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.uid_col not in df.columns or args.dataset_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{args.uid_col}' and '{args.dataset_col}'")

    # Work only on rows with the target dataset
    mask_target = df[args.dataset_col] == args.dataset
    df_tgt = df[mask_target].copy()

    # Find groups by (uid, dataset) and keep those that form pairs (size >= 2).
    # We'll remove exactly the first two occurrences per group in original order.
    # Determine the first index each group appears to preserve CSV order.
    df_tgt["_row_idx"] = df_tgt.index
    grp = df_tgt.groupby([args.uid_col, args.dataset_col])

    # Candidate groups: at least 2 rows
    candidate = (
        grp["_row_idx"]
        .agg(list)  # list of row indices (preserving order)
        .reset_index()
    )
    # Keep only groups with size >= 2
    candidate["len"] = candidate["_row_idx"].apply(len)
    candidate = candidate[candidate["len"] >= 2].copy()

    # Sort groups by first appearance in the CSV
    candidate["first_idx"] = candidate["_row_idx"].apply(lambda lst: min(lst))
    candidate = candidate.sort_values("first_idx")

    # Select up to N groups
    selected = candidate.head(args.pairs)

    # For each selected group, take the first two occurrences (the "pair")
    rows_to_drop = []
    for lst in selected["_row_idx"]:
        rows_to_drop.extend(lst[:2])  # drop only two rows per group

    # Build final DataFrame: drop those rows
    df_out = df.drop(index=rows_to_drop)

    # Save
    df_out.to_csv(args.output_csv, index=False)

    # Report
    print(f"Requested to drop {args.pairs} pairs for dataset='{args.dataset}'.")
    print(f"Dropped rows: {len(rows_to_drop)} (i.e., {len(rows_to_drop)//2} pairs).")
    print(f"Remaining rows: {len(df_out)}")
    print(f"Wrote: {args.output_csv}")

if __name__ == "__main__":
    main()
