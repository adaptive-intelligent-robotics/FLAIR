import argparse
import os
import time
import traceback
from typing import Dict, List, Tuple, Any
from itertools import cycle
from qdax.utils.plotting import plot_2d_map_elites_repertoire

import jax.numpy as jnp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MARKER_STYLES = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "H"]

#########
# Input #

parser = argparse.ArgumentParser()

# Folder
parser.add_argument("--results", default="map_elites_map", type=str)
parser.add_argument("--plots", default="map_elites_plots", type=str)

# Params
parser.add_argument("--legend-columns", default=1, type=int)
parser.add_argument("--legend-bottom", default=0.1, type=float)

# Process inputs
args = parser.parse_args()
save_folder = args.results
plot_folder = args.plots
assert os.path.exists(save_folder), "\n!!!ERROR!!! Empty result folder.\n"

################
# Get results #

# Create results folder if needed
try:
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)
except Exception:
    print("\n!!!WARNING!!! Cannot create folders for plots.")
    traceback.print_exc()

def load_results(
    save_folder: str,
    plot_folder: str,
) -> Tuple:
    """Main function to aload all results."""

    ############################
    # 1. Find all config files #

    # Open all config files in the folder
    print("\n\nOpening config files")
    folders = [
        root
        for root, dirs, files in os.walk(save_folder)
        for name in files
        if "config.csv" in name
    ]
    assert len(folders) > 0, "\n!!!ERROR!!! No config files in result folder.\n"
    config_frame = pd.DataFrame()
    for folder in folders:
        config_file = os.path.join(folder, "config.csv")
        sub_config_frame = pd.read_csv(config_file, index_col=False)
        sub_config_frame["folder"] = folder
        config_frame = pd.concat(
            [config_frame, sub_config_frame], ignore_index=True
        )
    assert (
        config_frame.shape[0] != 0
    ), "\n!!!ERROR!!! No runs refered in config files.\n"

    # Name algorithms
    print("\nSetting up algorithms names")
    use_in_name: Dict = {}
    for name_algo in config_frame["name"].drop_duplicates().values:
        sub_config_frame = config_frame[config_frame["name"] == name_algo]
        use_in_name[name_algo] = []
        for column in sub_config_frame.columns:
            all_values = sub_config_frame[column]
            all_values = all_values[all_values == all_values]  # remove NaN
            if not all_values.empty and all_values.nunique() > 1:
                use_in_name[name_algo].append(column)
    print("\n    Differences between runs:", use_in_name)

    # Add algo name to each line
    algos = []
    algos_batch = []
    for line in range(config_frame.shape[0]):
        algo = config_frame["name"][line]

        for name in use_in_name[config_frame["name"][line]]:
            if config_frame[name][line] == config_frame[name][line]:
                algo += " " + name + ":" + str(config_frame[name][line])

        algo_batch = algo + " - " + str(config_frame["batch_size"][line])
        algos.append(algo)
        algos_batch.append(algo_batch)

    config_frame["algo"] = algos
    config_frame["algo_batch"] = algos_batch
    config_frame = config_frame.reset_index(drop=True)

    print("    Found", config_frame.shape[0], "runs, with algo names:")
    print(config_frame["name"].drop_duplicates().reset_index(drop=True))
    print(config_frame["algo"].drop_duplicates().reset_index(drop=True))
    print(config_frame)

    ###########################
    # 2. Fill in replications #

    print("\nReading replications")
    start_t = time.time()

    replications_frame = pd.DataFrame(
        columns=["env_name", "algo", "batch_size", "num_rep"]
    )

    # Initialise replications_frame
    for env_name in config_frame["env_name"].drop_duplicates().values:
        for size in config_frame["batch_size"].drop_duplicates().values:
            for algo in config_frame["algo"].drop_duplicates().values:
                replications_frame = pd.concat(
                    [
                        replications_frame,
                        pd.DataFrame.from_dict(
                            {
                                "env_name": [env_name],
                                "algo": [algo],
                                "batch_size": [size],
                                "num_rep": [0],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

    # Go through all metrics files to fill replications_frame
    seeds: Dict = {}
    for line in range(config_frame.shape[0]):
        try:
            # Get the config for this line
            env_name = config_frame["env_name"][line]
            size = config_frame["batch_size"][line]
            name = config_frame["name"][line]
            algo = config_frame["algo"][line]
            seed = config_frame["seed"][line]

            if seed in seeds.keys():
                if seeds[seed][0] == env_name and seeds[seed][1] == name:
                    if seeds[seed][2] == size:
                        print(f"Twice seed for {env_name}, {name} and {size}.")
                    else:
                        print(f"Twice seed for {env_name} and {name}.")
                        print(f"Size are {seeds[seed][2]} and {size} respectively.")
            seeds[seed] = [env_name, name, size]

            # Add replication to frame
            replications_frame.loc[
                (replications_frame["env_name"] == env_name)
                & (replications_frame["algo"] == algo)
                & (replications_frame["batch_size"] == size),
                "num_rep",
            ] += 1

        except Exception:
            print("\n!!!WARNING!!! Cannot read line", line, ":")
            print(config_frame.loc[line])
            traceback.print_exc()

    # Remove empty replications from replications_frame
    replications_frame = replications_frame[replications_frame["num_rep"] != 0]

    # Save replications frame as csv
    print("\nReplications:")
    print(replications_frame)
    print("\n")
    replications_frame.to_csv(
        f"{plot_folder}/replications_frame.csv",
        index=None,
        sep=",",
    )

    print("Time to read replications:", time.time() - start_t)

    ################
    # 3. Load data #

    print("\nReading data")

    # Create the metrics dataframe
    all_convergence = pd.DataFrame()

    # Go through all metrics files
    start_t = time.time()
    rep = 0
    for line in range(config_frame.shape[0]):

        try:
            # First, get the config for this line
            name = config_frame["name"][line]
            algo = config_frame["algo"][line]
            algo_batch = config_frame["algo_batch"][line]
            env_name = config_frame["env_name"][line]
            size = config_frame["batch_size"][line]
            batch_size = config_frame["batch_size"][line]

            folder = config_frame["folder"][line]
            metrics_file = config_frame["metrics_file"][line]
            metrics_file = metrics_file[metrics_file.rfind("/") + 1 :]
            metrics_file = os.path.join(folder, metrics_file)

            # Second, read metrics
            data = pd.read_csv(metrics_file, index_col=False)
            if data.empty:
                print(f"!!!WARNING!!! {metrics_file} is empty.")
                continue

            # Third, merge everything in main dataframe
            data["name"] = name
            data["algo"] = algo
            data["algo_batch"] = algo_batch
            data["env_name"] = env_name
            data["batch_size"] = size
            data["batch_size"] = batch_size
            data["rep"] = rep

            all_convergence = pd.concat([all_convergence, data], ignore_index=True)

            # Increment rep counter
            rep += 1

        except Exception:
            print("\n!!!WARNING!!! Cannot read", metrics_file, ".")
            traceback.print_exc()

    print("Time to read all metrics frames:", time.time() - start_t)
    start_t = time.time()

    return config_frame, all_convergence

# Load everything
config_frame, all_convergence = load_results(
    save_folder=save_folder,
    plot_folder=plot_folder,
)

# Create a color frame
labels = config_frame["algo"].drop_duplicates().values
colors = sns.color_palette("colorblind", len(labels))
color_frame = pd.DataFrame(data={"Label": labels, "Color": colors})


######################
# Load min_max_frame #

def get_folder_name(config_frame: pd.DataFrame, name: str, line: int) -> str:
    folder = config_frame["folder"][line]
    folder_name = config_frame[name][line]
    if folder_name.rfind("/") == len(folder_name) - 1:
        folder_name = folder_name[:-1]
    folder_name = folder_name[folder_name.rfind("/") + 1 :]
    folder_name = os.path.join(folder, folder_name)
    return folder_name  # type: ignore


def find_min_max(
    plot_folder: str,
    config_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Function to find all the min and max for archives."""

    min_max_frame = pd.DataFrame()

    # Function used later on
    def _new_min_max(
        frame: str, name: str, line: int, min_values: float, max_values: float
    ) -> Tuple:
        folder = get_folder_name(frame, name, line)
        values = jnp.load(os.path.join(folder, "fitnesses.npy"))
        values_inf = jnp.where(values == -jnp.inf, jnp.inf, values)
        min_values = min(min_values, float(jnp.min(values_inf)))
        max_values = max(max_values, float(jnp.max(values)))
        return min_values, max_values

    # For each environment
    for env_name in config_frame["env_name"].drop_duplicates().values:

        env_config_frame = config_frame[(config_frame["env_name"] == env_name)].reset_index(
            drop=True
        )

        # Initialising all min and max
        min_fitness = jnp.inf
        max_fitness = -jnp.inf

        # For each run of this environment
        for line in range(env_config_frame.shape[0]):

            # Update min_fitness and max_fitness
            try:
                min_fitness, max_fitness = _new_min_max(
                    env_config_frame,
                    "results_repertoire",
                    line,
                    min_fitness,
                    max_fitness,
                )
            except Exception:
                print("!!!WARNING!!! Cannot extract min and max fitness.")
                traceback.print_exc()

        # Update the frame for this env
        min_max_frame = pd.concat(
            [
                min_max_frame,
                pd.DataFrame.from_dict(
                    {
                        "env_name": [env_name],
                        "min_fitness": min_fitness,
                        "max_fitness": max_fitness,
                    }
                ),
            ],
            ignore_index=True,
        )

    return min_max_frame

try:
    min_max_frame = find_min_max(
        plot_folder=plot_folder,
        config_frame=config_frame,
    )
except Exception:
    print("\n!!!WARNING!!! Cannot get min and max, not using normalisation.")
    min_max_frame = None
    traceback.print_exc()


#####################
# Plot main metrics #

def customize_axis(ax: Any) -> Any:
    """
    Customise axis for plots.
    """

    # Remove unused axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis="y", length=0)

    # Offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))

    # Put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle="--", linewidth=1.5)

    return ax

def plot_convergence(
    plot_folder: str,
    all_convergence: pd.DataFrame,
    color_frame: pd.DataFrame,
    x_column: str,
    x_name: str,
    legend_columns: int,
    legend_bottom: float,
) -> None:

    metrics = [
        f"qd_score",
        f"coverage",
        f"max_fitness",
    ]
    metrics_name = [
        f"QD-Score",
        f"Coverage",
        f"Max-Fitness",
    ]

    for idx in range(len(metrics)):

        try:
            file_name = (
                f"{plot_folder}/metrics_{metrics[idx]}.svg"
            )

            # Extract the list of environments
            env_name_lists = all_convergence["env_name"].drop_duplicates().values

            # Create figure
            nrows = 1
            ncols = len(env_name_lists)
            figsize = (ncols * 8, nrows * 6)
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=figsize, #sharey="col"
            )

            # Set palette and marker in common
            hue_values = all_convergence["algo"].drop_duplicates().values
            sub_color_frame = color_frame[color_frame["Label"].isin(hue_values)]
            env_palette = dict(zip(sub_color_frame["Label"], sub_color_frame["Color"]))
            markers = {
                hue_value: marker
                for hue_value, marker in zip(hue_values, cycle(MARKER_STYLES))
            }

            # Plot all subplots
            all_handles: List = []
            all_labels: List = []
            for nrow in range(nrows):

                for ncol in range(ncols):

                    # Get the data for this row and column
                    env_name = env_name_lists[ncol]
                    row_column_all_convergence = all_convergence[
                        all_convergence["env_name"] == env_name
                    ]

                    # Get axis
                    if nrows == 1 and ncols == 1:
                        ax = axes
                    elif nrows == 1:
                        ax = axes[ncol]
                    elif ncols == 1:
                        ax = axes[nrow]
                    else:
                        ax = axes[nrow, ncol]

                    sns.lineplot(
                        x=x_column,
                        y=metrics[idx],
                        data=row_column_all_convergence,
                        hue="algo",
                        estimator=np.median, 
                        errorbar=("pi", 50), 
                        style="algo",
                        ax=ax,
                        markers=markers,
                        dashes=not (markers),
                        palette=env_palette,
                    )
                    ax.set_xlabel(x_name, labelpad=7)
                    if ncol == 0:
                        ax.set_ylabel(metrics_name[idx], labelpad=7)
                    else:
                        ax.set_ylabel(None)
                    customize_axis(ax)
                    if nrow == 0:
                        ax.set_title(env_name)

                    if nrow != nrows - 1:
                        ax.get_xaxis().set_visible(False)

                    # Handle legends
                    handles, labels = ax.get_legend_handles_labels()
                    for i in range(len(labels)):
                        if labels[i] not in all_labels:
                            all_handles.append(handles[i])
                            all_labels.append(labels[i])
                    try:
                        ax.legend_.remove()
                    except Exception:
                        traceback.print_exc()

                    # Set y-axis in scientifix notation
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            # Tight layout
            plt.tight_layout(h_pad=1.70)

            # Add legend below graph
            fig.subplots_adjust(bottom=legend_bottom)
            fig.legend(
                handles=all_handles,
                labels=all_labels,
                loc="lower center",
                frameon=False,
                ncol=legend_columns,
            )

            # Save figure
            plt.savefig(file_name)

        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot metrics.")
            traceback.print_exc()

print("\nPlotting main metrics")
try:
    plot_convergence(
        plot_folder=plot_folder,
        all_convergence=all_convergence,
        color_frame=color_frame,
        x_column="epoch",
        x_name="Generations",
        legend_columns=args.legend_columns,
        legend_bottom=args.legend_bottom,
    )
except Exception:
    print("\n!!!WARNING!!! Cannot plot main convergence metrics.")
    traceback.print_exc()


#####################
# Plot all archives #

def plot_all_archives(
    plot_folder: str,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
) -> None:

    # For each environment
    for env_name in config_frame["env_name"].drop_duplicates().values:

        print(f"    Plotting for env_name {env_name}")
        env_config_frame = config_frame[(config_frame["env_name"] == env_name)].reset_index(
            drop=True
        )

        # Get all the corresponding min and max
        min_fitness = None
        max_fitness = None
        if min_max_frame is not None:
            env_min_max_frame = min_max_frame[min_max_frame["env_name"] == env_name]
            if not env_min_max_frame.empty:
                min_fitness = env_min_max_frame["min_fitness"][0]
                max_fitness = env_min_max_frame["max_fitness"][0]
        min_bd = [0, 0]
        if env_config_frame["min_bd"].values[0] != []:
            if "_" in env_config_frame["min_bd"].values[0]:
                min_bd_list = str(env_config_frame["min_bd"].values[0]).split("_")
            else:
                min_bd_list = str(env_config_frame["min_bd"].values[0])[1:-1].split(" ")
            if "" in min_bd_list:
                min_bd_list.remove("")
            min_bd = [float(bd) for bd in min_bd_list]
        max_bd = [1, 1]
        if env_config_frame["max_bd"].values[0] != []:
            if "_" in env_config_frame["max_bd"].values[0]:
                max_bd_list = str(env_config_frame["max_bd"].values[0]).split("_")
            else:
                max_bd_list = str(env_config_frame["max_bd"].values[0])[1:-1].split(" ")
            if "" in max_bd_list:
                max_bd_list.remove("")
            max_bd = [float(bd) for bd in max_bd_list]

        # For each run for this env
        for line in range(env_config_frame.shape[0]):

            # Create the figure
            algo = env_config_frame["algo"][line].replace(" ", "_")
            size = env_config_frame["batch_size"][line]
            file_name = f"{plot_folder}/{env_name}_{algo}_{size}_{line}_archive.png"
            ncols = 1
            nrows = 1
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8), sharey=True)

            try:
                results_repertoire = get_folder_name(env_config_frame, "results_repertoire", line)
                fitnesses = jnp.load(os.path.join(results_repertoire, "fitnesses.npy"))
                descriptors = jnp.load(
                    os.path.join(results_repertoire, "descriptors.npy")
                )
                centroids = jnp.load(os.path.join(results_repertoire, "centroids.npy"))

                if ncols == 1 and nrows == 1:
                    axes = ax
                else:
                    axes = ax.flat[0]

                _, _ = plot_2d_map_elites_repertoire(
                    centroids=centroids,
                    repertoire_fitnesses=fitnesses,
                    minval=min_bd,
                    maxval=max_bd,
                    vmin=min_fitness,
                    vmax=max_fitness,
                    repertoire_descriptors=descriptors,
                    ax=axes,
                )
                axes.set_title(f"{env_name} - {algo} - Original archive")

            except Exception:
                print("\n!!!WARNING!!! Cannot open repertoire for:")
                print(env_config_frame.loc[line])
                traceback.print_exc()

            # Finish figure
            plt.tight_layout()
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()

print("\nPlotting all archives")
try:
    plot_all_archives(
        plot_folder=plot_folder,
        config_frame=config_frame,
        min_max_frame=min_max_frame,
    )
except Exception:
    print("\n!!!WARNING!!! Cannot plot all archives.")
    traceback.print_exc()

