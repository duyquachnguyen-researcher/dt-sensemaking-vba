"""Cluster participants into placement profiles for Q11.

Builds participant feature vectors from stable seed statements, standardizes
them, selects k with gap statistic plus silhouette, then clusters via k-means.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
)

GRID_MIN = 1
GRID_MAX = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q11 participant profile clustering")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Placement.csv"),
        help="Path to Placement.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/q11_participant_profiles.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/q11_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Target number of clusters (defaults to gap statistic selection)",
    )
    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=0.8,
        help="Minimum participant coverage for stable statements",
    )
    parser.add_argument(
        "--gap-b",
        type=int,
        default=200,
        help="Number of reference datasets for gap statistic",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=6,
        help="Maximum clusters to evaluate (capped by participant count)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for k-means and reference datasets",
    )
    return parser.parse_args()


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        pass
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def deduplicate_latest(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], tuple[tuple[bool, datetime | None, int], dict[str, str]]] = {}
    for index, row in enumerate(rows):
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        ts_value = parse_timestamp(row.get("ts"))
        key = (token, statement_id)
        sort_key = (ts_value is not None, ts_value, index)
        if key not in latest or sort_key > latest[key][0]:
            latest[key] = (sort_key, row)
    return [item[1] for item in latest.values()]


def cleaned_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        token = (row.get("token") or "").strip()
        statement_id = (row.get("canonical_id") or "").strip()
        if not token or not statement_id:
            continue
        x_val = coerce_float(row.get("x"))
        y_val = coerce_float(row.get("y"))
        if x_val is None or y_val is None:
            continue
        if not (GRID_MIN <= x_val <= GRID_MAX and GRID_MIN <= y_val <= GRID_MAX):
            continue
        cleaned.append(
            {
                "token": token,
                "statement_id": statement_id,
                "x": x_val,
                "y": y_val,
            }
        )
    return cleaned


def build_participant_map(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, tuple[float, float]]]:
    participants: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        participants[row["token"]][row["statement_id"]] = (row["x"], row["y"])
    return participants


def stable_statement_ids(
    participants: dict[str, dict[str, tuple[float, float]]],
    threshold: float,
) -> list[str]:
    tokens = list(participants.keys())
    if not tokens:
        return []
    intersection = set(participants[tokens[0]].keys())
    for token in tokens[1:]:
        intersection &= set(participants[token].keys())
    if intersection:
        return sorted(intersection)
    counts: dict[str, int] = defaultdict(int)
    for placements in participants.values():
        for statement_id in placements:
            counts[statement_id] += 1
    min_count = math.ceil(len(tokens) * threshold)
    stable = [statement_id for statement_id, count in counts.items() if count >= min_count]
    if stable:
        return sorted(stable)
    return sorted(counts.keys())


def statement_centroids(
    participants: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, tuple[float, float]]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for placements in participants.values():
        for statement_id, (x_val, y_val) in placements.items():
            totals[statement_id][0] += x_val
            totals[statement_id][1] += y_val
            totals[statement_id][2] += 1
    centroids: dict[str, tuple[float, float]] = {}
    for statement_id, (x_sum, y_sum, count) in totals.items():
        if count:
            centroids[statement_id] = (x_sum / count, y_sum / count)
    return centroids


def build_feature_matrix(
    participants: dict[str, dict[str, tuple[float, float]]],
    statement_ids: list[str],
    centroids: dict[str, tuple[float, float]],
    neutral: tuple[float, float] = (4.0, 4.0),
) -> tuple[list[str], list[list[float]]]:
    tokens = sorted(participants.keys())
    matrix: list[list[float]] = []
    for token in tokens:
        placements = participants[token]
        row: list[float] = []
        for statement_id in statement_ids:
            if statement_id in placements:
                x_val, y_val = placements[statement_id]
            else:
                x_val, y_val = centroids.get(statement_id, neutral)
            row.extend([x_val, y_val])
        matrix.append(row)
    return tokens, matrix


def standardize(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    if not matrix:
        return [], [], []
    cols = len(matrix[0])
    means = []
    stds = []
    for col in range(cols):
        values = [row[col] for row in matrix]
        col_mean = mean(values)
        variance = mean([(value - col_mean) ** 2 for value in values])
        col_std = math.sqrt(variance)
        means.append(col_mean)
        stds.append(col_std)
    standardized = []
    for row in matrix:
        standardized.append(
            [
                (value - means[idx]) / stds[idx] if stds[idx] > 0 else 0.0
                for idx, value in enumerate(row)
            ]
        )
    return standardized, means, stds


def squared_distance(vec_a: list[float], vec_b: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))


def kmeans(
    matrix: list[list[float]],
    k: int,
    rng: random.Random,
    max_iter: int = 100,
    n_init: int = 10,
) -> tuple[list[int], list[list[float]], float]:
    if not matrix:
        return [], [], 0.0
    best_labels: list[int] = []
    best_centroids: list[list[float]] = []
    best_inertia = math.inf

    for _ in range(n_init):
        initial_indices = rng.sample(range(len(matrix)), k)
        centroids = [matrix[idx][:] for idx in initial_indices]
        labels = [0 for _ in matrix]
        for _ in range(max_iter):
            changed = False
            for i, row in enumerate(matrix):
                distances = [squared_distance(row, centroid) for centroid in centroids]
                new_label = distances.index(min(distances))
                if labels[i] != new_label:
                    labels[i] = new_label
                    changed = True
            new_centroids = [[0.0 for _ in centroids[0]] for _ in range(k)]
            counts = [0 for _ in range(k)]
            for label, row in zip(labels, matrix):
                counts[label] += 1
                for idx, value in enumerate(row):
                    new_centroids[label][idx] += value
            for idx in range(k):
                if counts[idx]:
                    new_centroids[idx] = [value / counts[idx] for value in new_centroids[idx]]
                else:
                    new_centroids[idx] = matrix[rng.randrange(len(matrix))][:]
            centroids = new_centroids
            if not changed:
                break
        inertia = sum(squared_distance(row, centroids[label]) for row, label in zip(matrix, labels))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels[:]
            best_centroids = [centroid[:] for centroid in centroids]
    return best_labels, best_centroids, best_inertia


def silhouette_score(matrix: list[list[float]], labels: list[int], k: int) -> float | None:
    if k < 2 or not matrix:
        return None
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    if len(clusters) < 2:
        return None
    scores = []
    for idx, row in enumerate(matrix):
        label = labels[idx]
        same_cluster = clusters[label]
        if len(same_cluster) < 2:
            continue
        a = mean(
            [
                math.sqrt(squared_distance(row, matrix[other]))
                for other in same_cluster
                if other != idx
            ]
        )
        b_values = []
        for other_label, indices in clusters.items():
            if other_label == label:
                continue
            b_values.append(
                mean([math.sqrt(squared_distance(row, matrix[other])) for other in indices])
            )
        if not b_values:
            continue
        b = min(b_values)
        scores.append((b - a) / max(a, b))
    if not scores:
        return None
    return mean(scores)


def reference_datasets(
    matrix: list[list[float]],
    rng: random.Random,
    b: int,
) -> list[list[list[float]]]:
    if not matrix:
        return []
    cols = len(matrix[0])
    mins = [min(row[col] for row in matrix) for col in range(cols)]
    maxs = [max(row[col] for row in matrix) for col in range(cols)]
    references = []
    for _ in range(b):
        sample = []
        for _ in matrix:
            row = [rng.uniform(mins[col], maxs[col]) for col in range(cols)]
            sample.append(row)
        references.append(sample)
    return references


def gap_statistic(
    matrix: list[list[float]],
    kmax: int,
    b: int,
    rng: random.Random,
) -> tuple[list[float], list[float], list[float]]:
    if not matrix:
        return [], [], []
    references = reference_datasets(matrix, rng, b)
    gaps = []
    s_k = []
    inertias = []
    for k in range(1, kmax + 1):
        _, _, inertia = kmeans(matrix, k, rng)
        inertias.append(inertia)
        log_wk = math.log(inertia) if inertia > 0 else -math.inf
        log_wkb = []
        for ref in references:
            _, _, ref_inertia = kmeans(ref, k, rng)
            log_wkb.append(math.log(ref_inertia) if ref_inertia > 0 else -math.inf)
        gap = mean(log_wkb) - log_wk
        gaps.append(gap)
        if b > 1:
            variance = mean([(val - mean(log_wkb)) ** 2 for val in log_wkb])
            sd = math.sqrt(variance)
        else:
            sd = 0.0
        s_k.append(math.sqrt(1 + 1 / b) * sd)
    return gaps, s_k, inertias


def select_k_gap(gaps: list[float], s_k: list[float]) -> int:
    for idx in range(len(gaps) - 1):
        if gaps[idx] >= gaps[idx + 1] - s_k[idx + 1]:
            return idx + 1
    if gaps:
        return gaps.index(max(gaps)) + 1
    return 1


def cluster_assignments(
    tokens: list[str],
    labels: list[int],
    participants: dict[str, dict[str, tuple[float, float]]],
    matrix: list[list[float]],
) -> list[dict[str, Any]]:
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_members[label].append(idx)
    results: list[dict[str, Any]] = []
    for idx, token in enumerate(tokens):
        label = labels[idx]
        members = cluster_members[label]
        distances_to_cluster = []
        for member_idx in members:
            if member_idx == idx:
                continue
            distances_to_cluster.append(
                math.sqrt(squared_distance(matrix[idx], matrix[member_idx]))
            )
        avg_cluster_distance = mean(distances_to_cluster) if distances_to_cluster else None
        results.append(
            {
                "token": token,
                "cluster_id": label + 1,
                "cluster_size": len(members),
                "placements": len(participants.get(token, {})),
                "avg_distance_to_cluster": avg_cluster_distance,
            }
        )
    return sorted(results, key=lambda row: (row["cluster_id"], row["token"]))


def summarize(
    stats: list[dict[str, Any]],
    statement_ids: list[str],
    gaps: list[float],
    s_k: list[float],
    silhouettes: dict[int, float],
    selected_k: int,
    gap_k: int,
    silhouette_k: int | None,
    kmax: int,
    raw_rows: int,
    dedup_rows: int,
    cleaned_rows_count: int,
) -> str:
    clusters = defaultdict(list)
    for row in stats:
        clusters[row["cluster_id"]].append(row)

    lines = [
        "Q11 Participant Profile Clustering Summary",
        "==========================================",
        f"Participants clustered: {len(stats)}",
        f"Clusters formed: {len(clusters)}",
        f"Stable statements used: {len(statement_ids)}",
        f"Kmax evaluated: {kmax}",
        f"Gap-selected k: {gap_k}",
        f"Silhouette-selected k: {silhouette_k}" if silhouette_k else "Silhouette-selected k: n/a",
        f"Final k used: {selected_k}",
        "",
        "Gap statistic results (k, gap, s_k)",
        "-----------------------------------",
    ]

    for idx, gap in enumerate(gaps, start=1):
        sk_value = s_k[idx - 1] if idx - 1 < len(s_k) else 0.0
        lines.append(f"k={idx}: gap={gap:.4f}, s_k={sk_value:.4f}")

    lines += [
        "",
        "Silhouette scores (k, score)",
        "-----------------------------",
    ]

    if silhouettes:
        for k in sorted(silhouettes):
            lines.append(f"k={k}: silhouette={silhouettes[k]:.4f}")
    else:
        lines.append("Silhouette scores not available (k<2).")

    lines += [
        "",
        "Cluster membership",
        "------------------",
    ]

    for cluster_id in sorted(clusters):
        members = sorted(clusters[cluster_id], key=lambda row: row["token"])
        lines.append(f"Cluster {cluster_id} (n={len(members)}):")
        for row in members:
            distance_text = (
                f"{row['avg_distance_to_cluster']:.2f}"
                if row["avg_distance_to_cluster"] is not None
                else "n/a"
            )
            lines.append(
                f"  - {row['token']}: placements={row['placements']}, avg_distance_to_cluster={distance_text}"
            )
        lines.append("")

    lines += [
        "Sanity checks",
        "-------------",
        f"Rows in input: {raw_rows}",
        f"Rows after dedup: {dedup_rows}",
        f"Rows after cleaning x/y: {cleaned_rows_count}",
    ]

    return "\n".join(lines)


def write_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token",
        "cluster_id",
        "cluster_size",
        "placements",
        "avg_distance_to_cluster",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    rows = read_rows(args.input)
    raw_rows = len(rows)
    deduped = deduplicate_latest(rows)
    dedup_rows = len(deduped)
    cleaned = cleaned_rows(deduped)

    participants = build_participant_map(cleaned)
    statement_ids = stable_statement_ids(participants, args.stable_threshold)
    centroids = statement_centroids(participants)
    tokens, matrix = build_feature_matrix(participants, statement_ids, centroids)
    standardized, _, _ = standardize(matrix)

    if not standardized:
        stats: list[dict[str, Any]] = []
        gaps: list[float] = []
        s_k: list[float] = []
        silhouettes: dict[int, float] = {}
        gap_k = 1
        silhouette_k = None
        selected_k = 1
        kmax = 1
    else:
        rng = random.Random(args.seed)
        kmax = min(args.kmax, max(len(tokens) - 1, 1))
        gaps, s_k, _ = gap_statistic(standardized, kmax, args.gap_b, rng)
        gap_k = select_k_gap(gaps, s_k)
        silhouettes = {}
        silhouette_k = None
        for k in range(2, kmax + 1):
            labels, _, _ = kmeans(standardized, k, rng)
            score = silhouette_score(standardized, labels, k)
            if score is not None:
                silhouettes[k] = score
        if silhouettes:
            silhouette_k = max(silhouettes, key=lambda key: silhouettes[key])
        if args.clusters:
            selected_k = args.clusters
        else:
            selected_k = gap_k
            if selected_k == 1 and silhouette_k:
                selected_k = silhouette_k
        labels, _, _ = kmeans(standardized, selected_k, rng)
        stats = cluster_assignments(tokens, labels, participants, standardized)

    write_csv(args.output_csv, stats)

    summary_text = summarize(
        stats=stats,
        statement_ids=statement_ids,
        gaps=gaps,
        s_k=s_k,
        silhouettes=silhouettes,
        selected_k=selected_k,
        gap_k=gap_k,
        silhouette_k=silhouette_k,
        kmax=kmax,
        raw_rows=raw_rows,
        dedup_rows=dedup_rows,
        cleaned_rows_count=len(cleaned),
    )

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)


if __name__ == "__main__":
    main()
