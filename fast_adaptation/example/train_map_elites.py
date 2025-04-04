import argparse
import os
import time
from functools import partial, reduce
from math import ceil
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import randint
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.map_elites import MAPElites
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.types import Metrics, RNGKey

from utils.metrics_manager import MetricsManager
from utils.set_up_hexapod import set_up_hexapod

# Limit CPU usage for HPC
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=4"
)

# Uncomment this to debug if time is spent rejiting functions.
# import logging
# logging.basicConfig(level=logging.DEBUG)


############
# 0. Input #
############

parser = argparse.ArgumentParser()

# Result folder (can be same for multiple runs)
parser.add_argument("--results", default="map_elites_map", type=str)

# Seed (set to a random number if left to 0)
parser.add_argument("--seed", default=0, type=int, help="Sampled if 0.")

# Suffixe to add a note to a run if changes not visible in params
parser.add_argument("--name", default="MAP-Elites", type=str)

# Stopping criterion (required)
parser.add_argument("--num-generations", default=50000, type=int)

# Compare size (required)
parser.add_argument("--batch-size", default=2048, type=int)

# Environment
parser.add_argument("--env-name", default="hexapod_velocity", type=str)
parser.add_argument("--episode-length", default=250, type=int)

# Metrics log-period
parser.add_argument("--log-period", default=50, type=int)
parser.add_argument("--archive-log-period", default=500, type=int)

# Archive
parser.add_argument("--euclidean-centroids", action="store_true")
parser.add_argument("--euclidean-grid-shape", default="50_50", type=str)

# Mutation
parser.add_argument("--iso-sigma", default=0.005, type=float)  # 0.05
parser.add_argument("--line-sigma", default=0.05, type=float)  # 0.1

args = parser.parse_args()


####################
# I. Configuration #
####################

print("\n\nEntering initialisation.\n")
step_t = time.time()

print("\n  -> Parameters processing.")

assert args.batch_size != 0, "\n!!!ERROR!!! No --batch-size."
assert args.num_generations != 0, "\n!!!ERROR!!! No --num-generations."

# Set random seed
args.seed = randint(1000000) if args.seed == 0 else args.seed

# Process grid structure
grid_shape = tuple([int(x) for x in args.euclidean_grid_shape.split("_")])
print(f"Using euclidean centroids with grid-shape {grid_shape}.")

print(f"    Using log_period {args.log_period}.")
print(f"    Using archive log_period {args.archive_log_period}.")

######################
# II. Initialisation #
######################

print("\n  -> Environment initialisation.")

# Init a random key
np.random.seed(args.seed)
random_key = jax.random.PRNGKey(args.seed)

# Set up the environment
(
    env,
    scoring_fn,
    init_policies_fn,
    policy_structure,
    min_bd,
    max_bd,
    qd_offset,
    random_key,
) = set_up_hexapod(
    env_name=args.env_name,
    episode_length=args.episode_length,
    batch_size=args.batch_size,
    random_key=random_key,
)
args.min_bd = min_bd
args.max_bd = max_bd
args.qd_offset = qd_offset

# Sample a set of initial solutions
init_policies, random_key = init_policies_fn(args.batch_size, random_key)
print(f"    Observation size: {env.observation_size}.")
print(f"    Action size: {env.action_size}.")

# Set up the algo
print("\n  -> Algorithm initialisation.")

variation_fn = partial(
    isoline_variation,
    iso_sigma=args.iso_sigma,
    line_sigma=args.line_sigma,
    minval=None,
    maxval=None,
)
emitter = MixingEmitter(
    mutation_fn=None,
    variation_fn=variation_fn,
    variation_percentage=1.0,
    batch_size=args.batch_size,
)

def empty_metrics_function(repertoire: Repertoire) -> Metrics:
    """Metrics are handled by the metrics manager."""
    return {}

map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=emitter,
    metrics_function=empty_metrics_function,
)

# Compute the centroids
print("\n  -> Centroid initialisation.")
centroids = compute_euclidean_centroids(
    grid_shape=grid_shape,
    minval=args.min_bd,
    maxval=args.max_bd,
)

# Set up the metric manager
print(f"\n  -> Metrics Manager initialisation.")
print(f"    Algorithm name {args.name}.")
metrics_manager = MetricsManager(
    args=args,
    name=args.name,
    scoring_fn=scoring_fn,
)

init_t = time.time() - step_t
print(f"\nFinished initialisation in {init_t} seconds.")

############
# III. Run #
############

print("\n\nEntering run.\n")
step_t = time.time()

# First iteration of the algorithm
repertoire, emitter_state, random_key = map_elites.init(
    init_policies, centroids, random_key
)
jax.tree_util.tree_map(
    lambda x: x.block_until_ready(), repertoire.genotypes
)  # ensure timing accuracy

# Initialise all counters
current_t = time.time() - step_t
total_metrics_t = 0.0
total_write_t = 0.0
epoch = 0
evals = args.batch_size

# Compute and write initial metrics
metrics_t, write_t = metrics_manager.write_all_metrics(
    epoch=epoch,
    evals=evals,
    current_time=current_t,
    repertoire=repertoire,
    emitter_state=emitter_state,
    random_key=random_key,
)
total_metrics_t += metrics_t
total_write_t += write_t

update_fn = jax.jit(map_elites.update)

# main loop
while epoch < args.num_generations:

    (
        repertoire,
        emitter_state,
        _,
        random_key,
    ) = update_fn(repertoire, emitter_state, random_key)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready(), repertoire.genotypes
    )  # ensure timing accuracy

    # Update all counters
    epoch += 1
    evals += args.batch_size

    # Check log period
    if epoch % args.log_period != 0:
        continue

    # Compute effective running time (remove all metric time)
    current_t = time.time() - step_t - total_write_t - total_metrics_t
    print(
        f"\n    Epoch: {epoch} / {args.num_generations}",
        f"-- Evals: {evals}",
        f"-- Time: {ceil(current_t)}",
        f"-- Total Time: {ceil(time.time() - step_t)}",
    )

    # Compute and write metrics
    metrics_t, write_t = metrics_manager.write_all_metrics(
        epoch=epoch,
        evals=evals,
        current_time=current_t,
        repertoire=repertoire,
        emitter_state=emitter_state,
        random_key=random_key,
    )
    total_metrics_t += metrics_t
    total_write_t += write_t


#################
# Final metrics #

# Compute effective running time (remove all metric time)
current_t = time.time() - step_t - total_write_t - total_metrics_t
print(
    f"\n    Ended at epoch: {epoch} / {args.num_generations}",
    f"-- Evals: {evals}",
    f"-- Time: {ceil(current_t)}",
    f"-- Total Time: {ceil(time.time() - step_t)}",
)

# Compute and write metrics
metrics_t, write_t = metrics_manager.write_all_metrics(
    epoch=epoch,
    evals=evals,
    current_time=current_t,
    repertoire=repertoire,
    emitter_state=emitter_state,
    random_key=random_key,
    final=True,
)

run_t = time.time() - step_t
print(f"\nFinished run in {run_t} seconds.")
