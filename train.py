#!/usr/bin/env python3

import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from core.data_preprocessing import DataPreprocessor
from core.domain_constraints import DomainConstraintBuilder
from models.mask_network import LearnableDomainMask, MaskOptimizer
from environments.causal_env import CausalEnvCF
from algorithms.vec2dag import Vec2DAGTransformer
from algorithms.consensus import ConsensusModule
from algorithms.counterfactual import CFAccConfig, CounterfactualAnalyzer
from configs.training_config import TrainingConfig
from utils.visualization import ResultVisualizer
from utils.logger import TrainingLogger

warnings.filterwarnings("ignore")


class CFMARLTrainer:
    def __init__(self, config: TrainingConfig):
        self._config = config
        self._setup_directories()
        self._setup_device()
        self._set_seeds()

        self._logger = TrainingLogger(
            log_dir=config.results_dir,
            tensorboard_dir=config.tensorboard_log
        )
        self._visualizer = ResultVisualizer()

        self._data = None
        self._encoders = None
        self._feature_names = None
        self._metadata = None
        self._mask_model = None
        self._mask_optimizer = None
        self._envs = []
        self._agents = []
        self._consensus_module = None
        self._vec2dag = None

    def _setup_directories(self):
        for dir_path in [
            self._config.checkpoint_dir,
            self._config.results_dir,
            self._config.tensorboard_log
        ]:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _setup_device(self):
        if self._config.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._config.device)

    def _set_seeds(self):
        np.random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config.seed)

    def load_data(self, data_path: str, feature_cols: Optional[List[str]] = None):
        preprocessor = DataPreprocessor(seed=self._config.seed)
        self._data, self._encoders, self._feature_names, self._metadata = (
            preprocessor.load_and_preprocess(
                data_path,
                feature_cols=feature_cols
            )
        )
        self._logger.log_info(f"Data loaded: {self._metadata}")

    def initialize_components(self):
        d = self._metadata["d"]

        constraint_builder = DomainConstraintBuilder(self._feature_names, d)
        C, E = constraint_builder.build_constraints()

        self._mask_model = LearnableDomainMask(
            d=d,
            C=C,
            E=E,
            device=str(self._device),
            seed=self._config.seed
        )

        self._mask_optimizer = MaskOptimizer(
            self._mask_model,
            learning_rate=self._config.mask.learning_rate,
            weight_decay=self._config.mask.weight_decay
        )

        cf_cfg = CFAccConfig(
            max_samples=self._config.counterfactual.max_samples,
            max_values_per_feature=self._config.counterfactual.max_values_per_feature,
            use_cache=self._config.counterfactual.use_cache,
            cache_size=self._config.counterfactual.cache_size,
            seed=self._config.seed
        )

        self._envs = []
        self._agents = []

        for k in range(self._config.num_agents):
            env = CausalEnvCF(
                d=d,
                data_cat=self._data,
                mask_model=self._mask_model,
                alpha=self._config.environment.alpha,
                beta=self._config.environment.beta,
                tau=self._config.environment.tau,
                z_clip=self._config.environment.z_clip,
                action_clip=self._config.environment.action_clip,
                cf_cfg=cf_cfg,
                cache_size=self._config.environment.cache_size,
                seed=self._config.seed + k
            )
            self._envs.append(env)

            agent = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=self._config.ppo.learning_rate,
                n_steps=self._config.ppo.n_steps,
                batch_size=self._config.ppo.batch_size,
                n_epochs=self._config.ppo.n_epochs,
                gamma=self._config.ppo.gamma,
                gae_lambda=self._config.ppo.gae_lambda,
                clip_range=self._config.ppo.clip_range,
                ent_coef=self._config.ppo.ent_coef,
                vf_coef=self._config.ppo.vf_coef,
                max_grad_norm=self._config.ppo.max_grad_norm,
                normalize_advantage=self._config.ppo.normalize_advantage,
                target_kl=self._config.ppo.target_kl,
                verbose=0,
                device=self._device,
                tensorboard_log=self._config.tensorboard_log
            )
            self._agents.append(agent)

        self._consensus_module = ConsensusModule(zeta=self._config.consensus.zeta)
        self._vec2dag = Vec2DAGTransformer(d=d, tau=self._config.environment.tau)

        self._logger.log_info("All components initialized")

    def train(self) -> Dict[str, Any]:
        history = []
        start_time = time.time()
        t = 0
        chunk_size = self._config.ppo.n_steps

        while t < self._config.total_timesteps:
            for agent in self._agents:
                agent.learn(
                    total_timesteps=chunk_size,
                    reset_num_timesteps=False,
                    progress_bar=False
                )
            t += chunk_size

            if (t % self._config.mask.update_frequency) == 0:
                loss = self._mask_optimizer.update(
                    lambda_prior=self._config.mask.lambda_prior,
                    num_steps=self._config.mask.update_steps
                )
                self._logger.log_scalar("mask_loss", loss, t)

            if self._config.consensus.enabled and (t % self._config.consensus.interval) == 0:
                metrics = self._consensus_module.update(self._envs, self._agents)
                self._logger.log_dict("consensus", metrics, t)

            if (t % self._config.eval_interval) == 0:
                eval_metrics = self._evaluate(t)
                history.append(eval_metrics)
                self._logger.log_dict("eval", eval_metrics, t)
                self._print_progress(t, eval_metrics, start_time)

            if (t % self._config.checkpoint_interval) == 0:
                self._save_checkpoint(t)

        final_adjacency = self._compute_final_adjacency()
        results = self._prepare_results(history, final_adjacency)
        self._save_results(results)

        return results

    def _evaluate(self, timestep: int) -> Dict[str, float]:
        with torch.no_grad():
            M = self._mask_model.mask_rollout().detach().cpu().numpy()

        adjacencies = []
        metrics_list = []

        for env in self._envs:
            A_soft = self._vec2dag.vec2dag_soft(env.z_state, M)
            A = self._vec2dag.binarize_adjacency(A_soft)

            if not self._vec2dag.is_cyclic(A):
                adjacencies.append(A)
                info = env._compute_reward_components(A)[1]
                metrics_list.append(info)

        if not metrics_list:
            return {
                "timestep": timestep,
                "dag_rate": 0.0,
                "avg_reward": -1e6,
                "avg_edges": 0.0,
                "avg_bic": float("inf"),
                "avg_cf_acc": 0.0
            }

        dag_rate = len(adjacencies) / self._config.num_agents
        avg_metrics = {
            key: float(np.mean([m[key] for m in metrics_list if key in m]))
            for key in ["bic", "cf_acc", "rsparse", "edges"]
        }

        avg_reward = sum(
            -avg_metrics["bic"] + avg_metrics["rsparse"] +
            self._config.environment.beta * avg_metrics["cf_acc"]
            for m in [avg_metrics]
        ) / len([avg_metrics])

        return {
            "timestep": timestep,
            "dag_rate": dag_rate,
            "avg_reward": avg_reward,
            "avg_edges": avg_metrics["edges"],
            "avg_bic": avg_metrics["bic"],
            "avg_cf_acc": avg_metrics["cf_acc"]
        }

    def _compute_final_adjacency(self) -> np.ndarray:
        with torch.no_grad():
            M = self._mask_model.mask_rollout().detach().cpu().numpy()

        adjacencies = []
        for env in self._envs:
            A_soft = self._vec2dag.vec2dag_soft(env.z_state, M)
            A = self._vec2dag.binarize_adjacency(A_soft)
            if not self._vec2dag.is_cyclic(A):
                adjacencies.append(A)

        if not adjacencies:
            d = self._metadata["d"]
            return np.zeros((d, d), dtype=np.int8)

        consensus = self._consensus_module.majority_vote_adjacency(
            adjacencies,
            threshold=self._config.consensus.majority_threshold
        )

        consensus = self._vec2dag.remove_cycles(consensus)
        return consensus

    def _prepare_results(
        self,
        history: List[Dict],
        final_adjacency: np.ndarray
    ) -> Dict[str, Any]:

        cf_analyzer = CounterfactualAnalyzer(
            CFAccConfig(seed=self._config.seed)
        )

        from algorithms.sem_estimator import SEMEstimator
        sem_estimator = SEMEstimator()

        bic, sems, topo = sem_estimator.compute_bic_and_sems(
            final_adjacency, self._data
        )

        cf_acc = cf_analyzer.compute_cf_acc(
            final_adjacency, self._data, sems, topo
        )

        attribution = cf_analyzer.compute_cf_attribution(
            final_adjacency, self._data, self._encoders,
            self._feature_names, sems, topo
        )

        return {
            "adjacency_matrix": final_adjacency,
            "history": pd.DataFrame(history),
            "feature_names": self._feature_names,
            "target_classes": self._encoders[-1].classes_.tolist(),
            "metadata": self._metadata,
            "final_bic": bic,
            "final_cf_acc": cf_acc,
            "attribution": attribution,
            "config": self._config.to_dict()
        }

    def _save_results(self, results: Dict[str, Any]):
        output_dir = Path(self._config.results_dir)

        np.save(output_dir / "adjacency_matrix.npy", results["adjacency_matrix"])
        results["history"].to_csv(output_dir / "training_history.csv", index=False)

        pd.DataFrame(
            results["attribution"],
            index=results["feature_names"],
            columns=results["target_classes"]
        ).to_csv(output_dir / "attribution_matrix.csv")

        import json
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "metadata": results["metadata"],
                    "final_bic": results["final_bic"],
                    "final_cf_acc": results["final_cf_acc"],
                    "config": results["config"]
                },
                f,
                indent=2
            )

    def _save_checkpoint(self, timestep: int):
        checkpoint_path = Path(self._config.checkpoint_dir) / f"checkpoint_{timestep}.pt"
        torch.save({
            "timestep": timestep,
            "mask_state_dict": self._mask_model.state_dict(),
            "mask_optimizer_state_dict": self._mask_optimizer.optimizer.state_dict(),
            "config": self._config.to_dict()
        }, checkpoint_path)
        self._logger.log_info(f"Checkpoint saved: {checkpoint_path}")

    def _print_progress(self, t: int, metrics: Dict, start_time: float):
        elapsed = (time.time() - start_time) / 60.0
        if self._config.verbose > 0:
            print(
                f"Step {t:7d} | {elapsed:6.1f}m | "
                f"R={metrics['avg_reward']:10.2f} | E={metrics['avg_edges']:6.1f} | "
                f"DAG={metrics['dag_rate']*100:5.1f}% | BIC={metrics['avg_bic']:10.1f} | "
                f"CF={metrics['avg_cf_acc']:6.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description="CF-MARL-SMV2D Training")
    parser.add_argument("--data", type=str, required=True, help="Path to data file")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--features", nargs="+", help="Feature columns to use")
    parser.add_argument("--timesteps", type=int, help="Total training timesteps")
    parser.add_argument("--agents", type=int, help="Number of agents")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.agents:
        config.num_agents = args.agents
    if args.seed:
        config.seed = args.seed

    config.validate()

    trainer = CFMARLTrainer(config)
    trainer.load_data(args.data, feature_cols=args.features)
    trainer.initialize_components()

    print("Starting CF-MARL-SMV2D training...")
    results = trainer.train()

    print("\n" + "="*50)
    print("Training completed!")
    print(f"Final adjacency edges: {results['adjacency_matrix'].sum()}")
    print(f"Final BIC: {results['final_bic']:.2f}")
    print(f"Final CF accuracy: {results['final_cf_acc']:.3f}")
    print(f"Results saved to: {config.results_dir}")
    print("="*50)


if __name__ == "__main__":
    main()