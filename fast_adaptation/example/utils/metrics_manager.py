import os
import csv
import time
from functools import partial
from typing import Any, Callable, Tuple, Dict

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Metrics, RNGKey

def save_config(save_folder: str, name: str, args: Any):
    """Save the current config in the config.csv file."""

    # Convert arguments to a dictionary
    args_dict = vars(args)
    args_dict["min_bd"] = "_".join(map(str, args_dict["min_bd"]))
    args_dict["max_bd"] = "_".join(map(str, args_dict["max_bd"]))

    # Add the name in first position
    args_dict = {**{"name": name}, **args_dict}

    # Create results folder if needed
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Opening config file and writing header
    file_name = f"{save_folder}/config.csv"
    if not os.path.exists(file_name):
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(args_dict.keys())

    # Writting config
    with open(file_name, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(args_dict.values())


def save_metrics(
    file_name: str,
    epoch: float,
    evals: float,
    time: float,
    metrics: Metrics,
    prefixe: str = "",
) -> None:
    """Save the current metrics in metric file."""

    def name(dic: Dict, name: str) -> Dict:
        return {f"{name}{key}": value for key, value in dic.items()}

    # Set name in all dictionaries
    metrics = name(metrics, prefixe)

    # Combine all of them
    all_metrics = {
        **metrics,
    }

    # Add epoch, eval, timestep and time
    all_metrics = {
        **{
            "epoch": epoch,
            "eval": evals,
            "time": time,
        },
        **all_metrics,
    }

    # Opening metric file and writing header
    if not os.path.exists(file_name):
        with open(file_name, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(all_metrics.keys())

    # Writting config
    with open(file_name, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(all_metrics.values())

class MetricsManager:
    """
    Class that contains all metrics computation and writting.
    """

    def __init__(
        self,
        args: Any,
        name: str,
        scoring_fn: Callable,
    ) -> None:
        """Set up all the necesary attributes and create all
        necessary folders and initialise all functions."""

        self._args = args
        self._name = name
        self._scoring_fn = scoring_fn

        # Create results folder
        if not os.path.exists(self._args.results):
            os.mkdir(self._args.results)

        # Create all the folders to save repertoires
        repertoire_suffixe = (
            "repertoire_" + self._name + "_" + str(self._args.seed) + "/"
        )
        self._args.results_repertoire = self._args.results + "/" + repertoire_suffixe
        if not os.path.exists(self._args.results_repertoire):
            os.mkdir(self._args.results_repertoire)

        # Create the metrics files
        self._args.metrics_file = (
            f"{self._args.results}/metrics_{self._name}_{str(self._args.seed)}.csv"
        )

        # Write the corresponding config
        save_config(save_folder=self._args.results, name=self._name, args=self._args)

    def write_all_metrics(
        self,
        epoch: int,
        evals: float,
        current_time: float,
        repertoire: Repertoire,
        emitter_state: Any,
        random_key: RNGKey,
        final: bool = False,
    ) -> Tuple[float, float]:
        """Main metric function. Compute and write all the metrics and
        all the repertoires.
        """
        metrics_t = 0.0
        write_t = 0.0

        # Compute metrics on the repertoire
        start_t = time.time()
        metrics = self._metrics_function(repertoire)
        metrics_t += time.time() - start_t

        # Sanity check of min_fitness
        start_t = time.time()
        fitnesses = repertoire.fitnesses
        fitnesses = jnp.where(fitnesses == -jnp.inf, jnp.inf, fitnesses)
        if jnp.min(fitnesses) < -self._args.qd_offset:
            print(
                f"!!!WARNING!!! got min fit {jnp.min(fitnesses)} < {-self._args.qd_offset},"
                "may lead to inacurate QD-Score."
            )
        metrics_t += time.time() - start_t

        # Write metrics
        start_t = time.time()
        save_metrics(
            file_name=self._args.metrics_file,
            epoch=epoch,
            evals=evals,
            time=current_time,
            metrics=metrics,
        )
        print("    -> Metrics saved in", self._args.metrics_file)

        # Write repertoire
        if epoch % self._args.archive_log_period == 0 or final:
            repertoire.save(path=self._args.results_repertoire)
            print("    -> Repertoire saved in", self._args.results_repertoire)

        write_t += time.time() - start_t

        # Return all the timings
        return metrics_t, write_t

    @partial(jax.jit, static_argnames=("self",))
    def _metrics_function(self, repertoire: Repertoire) -> Metrics:
        repertoire_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
        qd_score += self._args.qd_offset * jnp.sum(1.0 - repertoire_empty)
        coverage = 100 * jnp.mean(1.0 - repertoire_empty)
        max_fitness = jnp.max(repertoire.fitnesses)
        min_fitness = jnp.min(
            jnp.where(repertoire.fitnesses == -jnp.inf, jnp.inf, repertoire.fitnesses)
        )
        return {
            "qd_score": qd_score,
            "max_fitness": max_fitness,
            "min_fitness": min_fitness,
            "coverage": coverage,
        }
