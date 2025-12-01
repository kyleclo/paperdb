import json
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.isotonic import IsotonicRegression


def collect_metrics(results_dir: Path) -> dict:
    """Collect all metrics from results directory."""
    metrics_data = []

    # Find all metrics.json files
    for metrics_file in results_dir.rglob("*.metrics.json"):
        # Parse path: results/{dataset}/{split}.{method}.metrics.json
        relative_path = metrics_file.relative_to(results_dir)
        dataset = relative_path.parts[0]
        filename = relative_path.parts[-1]

        # Extract split and method from filename
        # Format: {split}.{method}.metrics.json or {split}.{method}.results.metrics.json
        filename_without_ext = filename.replace(".metrics.json", "")
        if filename_without_ext.endswith(".results"):
            filename_without_ext = filename_without_ext.replace(".results", "")

        parts = filename_without_ext.split(".", 1)
        if len(parts) == 2:
            split, method = parts
        else:
            continue  # Skip if can't parse

        # Load metrics
        with open(metrics_file) as f:
            metrics = json.load(f)

        metrics_data.append({
            "dataset": dataset,
            "split": split,
            "method": method,
            "hits@1": metrics.get("hits@1", 0.0),
            "hits@5": metrics.get("hits@5", 0.0),
            "mrr": metrics.get("mrr", 0.0),
            "total_queries": metrics.get("total_queries", 0),
        })

    return metrics_data


def create_markdown_report(metrics_data: list) -> str:
    """Create a markdown report summarizing all metrics."""
    if not metrics_data:
        return "No metrics found.\n"

    # Sort by dataset, split, then method
    metrics_data = sorted(metrics_data, key=lambda x: (x["dataset"], x["split"], x["method"]))

    # Create markdown table
    lines = ["# Evaluation Results Summary\n"]
    lines.append("| Dataset | Split | Method | Hits@1 | Hits@5 | MRR | Queries |")
    lines.append("|---------|-------|--------|--------|--------|-----|---------|")

    for row in metrics_data:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['method']} | "
            f"{row['hits@1']:.2%} | {row['hits@5']:.2%} | {row['mrr']:.3f} | "
            f"{row['total_queries']} |"
        )

    lines.append("")

    # Add summary statistics by method
    lines.append("## Summary by Method\n")
    method_stats = defaultdict(lambda: {"hits@1": [], "hits@5": [], "mrr": []})

    for row in metrics_data:
        method_stats[row["method"]]["hits@1"].append(row["hits@1"])
        method_stats[row["method"]]["hits@5"].append(row["hits@5"])
        method_stats[row["method"]]["mrr"].append(row["mrr"])

    lines.append("| Method | Avg Hits@1 | Avg Hits@5 | Avg MRR | Runs |")
    lines.append("|--------|------------|------------|---------|------|")

    for method in sorted(method_stats.keys()):
        stats = method_stats[method]
        avg_hits1 = sum(stats["hits@1"]) / len(stats["hits@1"])
        avg_hits5 = sum(stats["hits@5"]) / len(stats["hits@5"])
        avg_mrr = sum(stats["mrr"]) / len(stats["mrr"])
        lines.append(
            f"| {method} | {avg_hits1:.2%} | {avg_hits5:.2%} | "
            f"{avg_mrr:.3f} | {len(stats['hits@1'])} |"
        )

    lines.append("")

    # Add summary statistics by dataset
    lines.append("## Summary by Dataset\n")
    dataset_stats = defaultdict(lambda: {"hits@1": [], "hits@5": [], "mrr": []})

    for row in metrics_data:
        dataset_stats[row["dataset"]]["hits@1"].append(row["hits@1"])
        dataset_stats[row["dataset"]]["hits@5"].append(row["hits@5"])
        dataset_stats[row["dataset"]]["mrr"].append(row["mrr"])

    lines.append("| Dataset | Avg Hits@1 | Avg Hits@5 | Avg MRR | Runs |")
    lines.append("|---------|------------|------------|---------|------|")

    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        avg_hits1 = sum(stats["hits@1"]) / len(stats["hits@1"])
        avg_hits5 = sum(stats["hits@5"]) / len(stats["hits@5"])
        avg_mrr = sum(stats["mrr"]) / len(stats["mrr"])
        lines.append(
            f"| {dataset} | {avg_hits1:.2%} | {avg_hits5:.2%} | "
            f"{avg_mrr:.3f} | {len(stats['hits@1'])} |"
        )

    return "\n".join(lines) + "\n"


def parse_difficulty(dataset_name: str) -> tuple:
    """Parse td and md values from dataset name.

    Returns (td, md) tuple or (None, None) if not found.
    """
    # Pattern: metadata_as_query_td{value}_md{value}
    match = re.search(r'td([\d.]+)_md([\d.]+)', dataset_name)
    if match:
        td = float(match.group(1))
        md = float(match.group(2))
        return td, md
    return None, None


def create_query_type_comparison_plot(metrics_data: list, output_dir: Path):
    """Create plot comparing content_as_query vs metadata_as_query performance."""
    # Group by method and query type
    method_query_data = defaultdict(lambda: {"content": [], "metadata": []})

    for row in metrics_data:
        dataset = row["dataset"]
        method = row["method"]
        hits_at_1 = row["hits@1"] * 100  # Convert to percentage

        # Skip old versions of relational methods (keep only v2)
        if method == "relational-detailed" or method == "relational-minimal":
            continue

        if "content_as_query" in dataset:
            method_query_data[method]["content"].append(hits_at_1)
        elif "metadata_as_query" in dataset or dataset == "metadata_as_query":
            method_query_data[method]["metadata"].append(hits_at_1)

    # Calculate averages for each method and query type
    methods = sorted(method_query_data.keys())
    content_avgs = []
    metadata_avgs = []

    for method in methods:
        data = method_query_data[method]
        content_avg = np.mean(data["content"]) if data["content"] else 0
        metadata_avg = np.mean(data["metadata"]) if data["metadata"] else 0
        content_avgs.append(content_avg)
        metadata_avgs.append(metadata_avg)

    # Create grouped bar chart
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 8))
    bars1 = ax.bar(x - width/2, content_avgs, width, label='Content as Query', alpha=0.8)
    bars2 = ax.bar(x + width/2, metadata_avgs, width, label='Metadata as Query', alpha=0.8)

    ax.set_xlabel('Method', fontsize=18)
    ax.set_ylabel('Average Hits@1 (%)', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='best', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "query_type_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Query type comparison plot saved to: {output_path}")
    plt.close()


def create_dropout_plots(metrics_data: list, output_dir: Path):
    """Create separate plots for title dropout and metadata dropout effects with boxplots."""
    # Filter for metadata_as_query datasets with difficulty parameters
    metadata_results = []

    for row in metrics_data:
        dataset = row["dataset"]
        method = row["method"]

        # Skip old versions of relational methods (keep only v2)
        if method == "relational-detailed" or method == "relational-minimal":
            continue

        if "metadata_as_query" in dataset:
            td, md = parse_difficulty(dataset)
            if td is not None and md is not None:
                metadata_results.append({
                    "method": method,
                    "td": td,
                    "md": md,
                    "hits@1": row["hits@1"],
                    "hits@5": row["hits@5"],
                    "dataset": dataset
                })

    if not metadata_results:
        print("No metadata_as_query datasets with difficulty parameters found")
        return

    # Get unique dropout values and methods
    td_values = sorted(set(r["td"] for r in metadata_results))
    md_values = sorted(set(r["md"] for r in metadata_results))
    methods = sorted(set(r["method"] for r in metadata_results))

    # Color palette
    colors = {method: plt.cm.tab10(i) for i, method in enumerate(methods)}

    # Plot 1: Title Dropout Effect (showing distribution over MD values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for metric_idx, (metric_name, ax) in enumerate([("hits@1", ax1), ("hits@5", ax2)]):
        for method in methods:
            # Collect data for boxplots
            boxplot_data = []
            positions = []

            for i, td in enumerate(td_values):
                values = []
                for row in metadata_results:
                    if row["method"] == method and row["td"] == td:
                        values.append(row[metric_name] * 100)

                if values:
                    boxplot_data.append(values)
                    # Offset positions for each method
                    offset = (methods.index(method) - len(methods)/2 + 0.5) * 0.15
                    positions.append(i + offset)

            # Create boxplots
            bp = ax.boxplot(boxplot_data, positions=positions, widths=0.12,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=colors[method], alpha=0.7),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(color=colors[method]),
                           capprops=dict(color=colors[method]))

            # Add connecting line through medians
            medians = [np.median(data) for data in boxplot_data]
            ax.plot(positions, medians, color=colors[method], linewidth=2,
                   label=method, alpha=0.8, zorder=0)

        ax.set_xlabel("Title Dropout (TD)", fontsize=16)
        ax.set_ylabel(f"{metric_name.upper()} (%)", fontsize=16)
        ax.set_xticks(range(len(td_values)))
        ax.set_xticklabels([f"{td:.1f}" for td in td_values])
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Effect of Title Dropout on Performance", fontsize=20, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "title_dropout_vs_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Title dropout plot saved to: {output_path}")
    plt.close()

    # Plot 2: Metadata Dropout Effect (showing distribution over TD values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for metric_idx, (metric_name, ax) in enumerate([("hits@1", ax1), ("hits@5", ax2)]):
        for method in methods:
            # Collect data for boxplots
            boxplot_data = []
            positions = []

            for i, md in enumerate(md_values):
                values = []
                for row in metadata_results:
                    if row["method"] == method and row["md"] == md:
                        values.append(row[metric_name] * 100)

                if values:
                    boxplot_data.append(values)
                    # Offset positions for each method
                    offset = (methods.index(method) - len(methods)/2 + 0.5) * 0.15
                    positions.append(i + offset)

            # Create boxplots
            bp = ax.boxplot(boxplot_data, positions=positions, widths=0.12,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=colors[method], alpha=0.7),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(color=colors[method]),
                           capprops=dict(color=colors[method]))

            # Add connecting line through medians
            medians = [np.median(data) for data in boxplot_data]
            ax.plot(positions, medians, color=colors[method], linewidth=2,
                   label=method, alpha=0.8, zorder=0)

        ax.set_xlabel("Metadata Dropout (MD)", fontsize=16)
        ax.set_ylabel(f"{metric_name.upper()} (%)", fontsize=16)
        ax.set_xticks(range(len(md_values)))
        ax.set_xticklabels([f"{md:.1f}" for md in md_values])
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Effect of Metadata Dropout on Performance", fontsize=20, y=0.995)
    plt.tight_layout()

    output_path = output_dir / "metadata_dropout_vs_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metadata dropout plot saved to: {output_path}")
    plt.close()


def main():
    results_dir = Path(__file__).parent.parent / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print(f"Collecting metrics from: {results_dir}")
    metrics_data = collect_metrics(results_dir)

    print(f"Found {len(metrics_data)} metric files")

    # Generate markdown report
    report = create_markdown_report(metrics_data)

    # Write to file
    output_file = results_dir / "summary.md"
    with open(output_file, "w") as f:
        f.write(report)

    print(f"\nSummary written to: {output_file}")
    print("\n" + report)

    # Generate plots
    print("\nGenerating plots...")
    create_query_type_comparison_plot(metrics_data, results_dir)
    create_dropout_plots(metrics_data, results_dir)


if __name__ == "__main__":
    main()
