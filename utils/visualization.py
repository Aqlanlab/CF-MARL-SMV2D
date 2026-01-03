import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path


class ResultVisualizer:
    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        self._style = style
        self._figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_adjacency_matrix(
        self,
        adjacency: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:

        fig, ax = plt.subplots(figsize=self._figsize)

        labels = feature_names if feature_names else [f"X{i}" for i in range(adjacency.shape[0])]

        sns.heatmap(
            adjacency,
            xticklabels=labels,
            yticklabels=labels,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title("Discovered Causal Structure", fontsize=16, fontweight='bold')
        ax.set_xlabel("Effect Variables", fontsize=12)
        ax.set_ylabel("Cause Variables", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_causal_graph(
        self,
        adjacency: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        layout: str = "spring"
    ) -> None:

        G = nx.DiGraph(adjacency)

        if feature_names:
            mapping = {i: name for i, name in enumerate(feature_names)}
            G = nx.relabel_nodes(G, mapping)

        fig, ax = plt.subplots(figsize=self._figsize)

        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.shell_layout(G)

        node_colors = ['lightblue' if i < len(G) - 1 else 'lightcoral'
                      for i in range(len(G))]

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=2000,
            alpha=0.9,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>',
            width=2,
            alpha=0.6,
            ax=ax
        )

        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        ax.set_title("Causal DAG Structure", fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_history(
        self,
        history: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:

        if metrics is None:
            metrics = ["avg_reward", "avg_edges", "dag_rate", "avg_bic", "avg_cf_acc"]

        available_metrics = [m for m in metrics if m in history.columns]
        n_metrics = len(available_metrics)

        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            ax.plot(history["timestep"], history[metric], linewidth=2)
            ax.set_xlabel("Training Steps", fontsize=10)
            ax.set_ylabel(self._format_metric_name(metric), fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Training Progress: {self._format_metric_name(metric)}",
                        fontsize=12, fontweight='bold')

        plt.suptitle("CF-MARL-SMV2D Training History", fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_attribution_heatmap(
        self,
        attribution: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        save_path: Optional[str] = None
    ) -> None:

        fig, ax = plt.subplots(figsize=(14, 10))

        sns.heatmap(
            attribution,
            xticklabels=class_names,
            yticklabels=feature_names,
            cmap="RdBu_r",
            center=0,
            vmin=-15,
            vmax=8,
            square=False,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Causal Attribution Score"},
            ax=ax
        )

        ax.set_title("Counterfactual Attribution Analysis", fontsize=16, fontweight='bold')
        ax.set_xlabel("Target Classes", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_consensus_evolution(
        self,
        consensus_history: List[Dict],
        save_path: Optional[str] = None
    ) -> None:

        if not consensus_history:
            return

        df = pd.DataFrame(consensus_history)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self._figsize)

        ax1.plot(df.index, df["z_variance"], linewidth=2, label="Z-state variance")
        ax1.set_xlabel("Consensus Updates", fontsize=10)
        ax1.set_ylabel("Variance", fontsize=10)
        ax1.set_title("Agent State Convergence", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(df.index, df["param_variance"], linewidth=2, label="Parameter variance", color='orange')
        ax2.set_xlabel("Consensus Updates", fontsize=10)
        ax2.set_ylabel("Variance", fontsize=10)
        ax2.set_title("Policy Parameter Convergence", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.suptitle("Multi-Agent Consensus Evolution", fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_summary_report(
        self,
        results: Dict[str, Any],
        output_dir: str
    ) -> None:

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.plot_adjacency_matrix(
            results["adjacency_matrix"],
            results["feature_names"],
            save_path=str(output_path / "adjacency_matrix.png")
        )

        self.plot_causal_graph(
            results["adjacency_matrix"],
            results["feature_names"],
            save_path=str(output_path / "causal_graph.png")
        )

        if "history" in results and isinstance(results["history"], pd.DataFrame):
            self.plot_training_history(
                results["history"],
                save_path=str(output_path / "training_history.png")
            )

        if "attribution" in results:
            self.plot_attribution_heatmap(
                results["attribution"],
                results["feature_names"],
                results["target_classes"],
                save_path=str(output_path / "attribution_heatmap.png")
            )

    def _format_metric_name(self, metric: str) -> str:
        replacements = {
            "avg_": "Average ",
            "_": " ",
            "cf": "CF",
            "acc": "Accuracy",
            "bic": "BIC",
            "dag": "DAG"
        }
        name = metric
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name.title()