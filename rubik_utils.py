# %% [code]
import pandas as pd
import numpy as np
import polars as pl
import torch
from time import time
import json
from sympy.combinatorics.permutations import Permutation
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from typing import Iterable


# God's number (in QTM) for Pocket Cube
N_GOD = 14

# The number of all states for Pocket Cube
N_STATES = 3674160


def get_moves(puzzle_type: str, filepath: str) -> dict[str, list[int]]:
    """
    Generates moves in the format `move name` -> `permutation of states`.

    Args:
        puzzle_type: A string description of the puzzle (e.g., "cube_2/2/2", "cube_3/3/3", etc)

    Returns:
        A mapping where each key is a move (as a string, e.g. "f0", "-r1") and each value 
        is a list of integers representing the permutation that defines the move.
    """
    puzzle_info_df = pd.read_csv(filepath)
    allowed_moves_str = puzzle_info_df.loc[puzzle_info_df['puzzle_type']
                                           == puzzle_type, 'allowed_moves'].iloc[0]
    allowed_moves_dict = json.loads(allowed_moves_str.replace("'", '"'))

    moves = {}
    for k, mv in allowed_moves_dict.items():
        moves[k] = mv

        # Add inverse moves
        moves["-" + k] = list(Permutation(mv) ** -1)

    return moves


def show_cube_moves(puzzle_type: str,
                    filepath: str, move: str,
                    show_numbers: bool = True,
                    save_figure: bool = False):
    """
    Show states on cube unfolding after making move
    """
    moves = get_moves(puzzle_type, filepath)
    state_size = len(moves[move])
    target = list(range(state_size))
    l = int(np.sqrt(state_size/6))
    grid_height = l*3
    grid_width = l*4

    xbases = [l, l, l*2, l*3, 0, l]
    ybases = [l*3, l*2, l*2, l*2, l*2, l]

    # colors = generate_gradient_colors(N)
    colors = ['lightgray', 'lightgreen', 'tomato',
              'cornflowerblue', 'orange', 'yellow']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, mi in zip(axes, [("before", target), ("after", moves[move])]):
        k, mv = mi

        ax.set_xlim([0, grid_width])
        ax.set_ylim([0, grid_height])
        ax.set_xticks(range(grid_width))
        ax.set_yticks(range(grid_height))

        for face in range(6):
            for i in range(l*l):
                dx, dy = i % l, i//l
                x = xbases[face] + dx
                y = ybases[face] - dy - 1

                ii = face*l*l + i
                c = colors[mv[ii]//(l*l)]
                # c = colors[mv[ii]]

                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=c))
                if show_numbers:
                    ax.text(x + 0.5, y + 0.5,
                            mv[ii], ha='center', va='center', color='black')

        ax.grid(True)
        ax.set_title(k)
        ax.tick_params(labelbottom=False, labelleft=False,
                       labelright=False, labeltop=False)
    if save_figure:
        plt.savefig(f"{move}.svg")
    plt.show()


def row_difference(A, B: torch.tensor) -> torch.tensor:
    r"""
        Calulates A \ B — set difference between rows of A and rows of B
    """
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    return A[~mask[:len(A)]]


class TensorCube:
    def __init__(self, device, puzzle_type: str = "cube_2/2/2",
                 generators: list[str] = [
                     'f0', '-f0', 'r0', '-r0', 'd0', '-d0'],
                 verbose: bool = True):
        self.device = device
        self.puzzle_type = puzzle_type
        self.generators = generators
        self.available_moves = self._moves2tensor()
        self.state_size = self.available_moves.shape[1]
        self.states = torch.zeros(
            N_STATES, self.state_size, device=self.device, dtype=torch.int8)
        self.distances = torch.zeros(
            N_STATES, dtype=torch.int8, device=self.device)
        self.layers_pop = np.zeros(N_GOD + 1, dtype=np.int32)
        self.visit_cube_states(verbose)
        self.neighbors_tensor = self._get_neighbors_tensor()

    @property
    def n_gens(self):
        return len(self.generators)

    def _moves2tensor(self):
        moves = get_moves(self.puzzle_type)
        return torch.tensor([moves[gen] for gen in self.generators], device=self.device)

    def get_neighbors(self, states, moves):
        """
        Some torch magic to calculate all new states which can be obtained from states by moves
        """
        return torch.gather(
            states.unsqueeze(1).expand(
                states.size(0), self.n_gens, states.size(1)),
            2,
            moves.unsqueeze(0).expand(states.size(0), self.n_gens, states.size(1)))

    def visit_cube_states(self, verbose: bool):
        """
        BFS over all possible states, calculating distances to the initial (target) state
        """
        # all_states[0, :state_size] = torch.arange(6, device=device, dtype=dtype_int).repeat_interleave(state_size//6)
        if verbose:
            print("Starting BFS...")
        start_time = time()
        self.states[0, :self.state_size] = torch.arange(
            self.state_size, device=self.device, dtype=torch.int8
        )
        states = self.states[0].unsqueeze(0)
        begin = 1
        self.layers_pop[0] = 1
        for j in range(N_GOD):
            states = torch.unique(
                self.get_neighbors(
                    states, self.available_moves).flatten(end_dim=1),
                sorted=False,
                dim=0)
            states = row_difference(states, self.states[:begin])
            end = begin + states.shape[0]
            self.states[begin:end, :] = states
            self.distances[begin:end] = j + 1
            begin = end
            self.layers_pop[j + 1] = states.shape[0]
            if verbose:
                print(f"layer {j + 1}:", states.shape)
        if verbose:
            print("BFS finised in {:.2f} s".format(time() - start_time))

    def _get_neighbors_tensor(self):
        schema = {f"col_{i}": pl.Int8 for i in range(self.state_size)}
        all_states_df = pl.DataFrame(
            self.states.cpu().numpy(), schema=schema).with_row_index()
        all_neighbors = self.get_neighbors(self.states, self.available_moves)
        all_neighbors_df = pl.DataFrame(
            all_neighbors.cpu().numpy().reshape(-1, 24), schema=schema)
        indexed_neighbors = all_states_df.join(
            all_neighbors_df, on=list(schema.keys()))
        neighbors_tensor = indexed_neighbors['index'].to_torch().view((-1, 6))
        del all_neighbors
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return neighbors_tensor

    def init_distances(self, init_type):
        cum_sums = self.layers_pop.cumsum()
        Y0 = torch.zeros((N_STATES,), dtype=torch.int, device=self.device)
        if isinstance(init_type, int):
            # assign first several layers by true distance
            assert init_type < len(cum_sums)
            Y0[:cum_sums[init_type]] = self.distances[:cum_sums[init_type]]
        elif isinstance(init_type, list):
            # assign each layer by a specific value
            assert len(init_type) == 14
            for i in range(14):
                Y0[cum_sums[i]:cum_sums[i + 1]] = init_type[i]
        elif init_type == "hamming":
            Y0 = torch.cdist(
                self.states.to(device=self.device, dtype=torch.float64),
                self.states[0].unsqueeze(0).to(device=self.device, dtype=torch.float64), p=0
            ).squeeze() * 14. / 24.
        elif init_type == "manhattan":
            norm = torch.linalg.vector_norm((self.states - self.states[0, :].unsqueeze(
                0)).to(device=self.device, dtype=torch.float64), 1, dim=-1)
            Y0 = norm * 14. / norm.max()
        return Y0

    def iterations_to_convergence(self,
                                  alphas: Iterable[int],
                                  init_type: int | str | list[int] | None = None,
                                  eps: float = 1e-5,
                                  max_iter: int = 1000,
                                  correlation_func=spearmanr,
                                  noise_begin: int = N_GOD,
                                  sigma: float = 1.0,
                                  verbose: bool = False):
        """
        Finds number of iterations and correlations on each iteration of DP algorithm

        Args:
            alphas: list of floats from (0, 1]
            init_type: specifies the initial values of distances (None = zero init)
            eps: float, tolerance
            max_iter: int, maximum number of iterations
            correlation_func: Callable, function which calculate correlation (Pearson or Spearman)
            noise_begin: int, layer from which gaussian random noise is added on each iteration (no noise by default)
        """
        cum_sums = self.layers_pop.cumsum()
        Y0 = self.init_distances(init_type)
        iterations = []
        correlations = []
        for alpha in alphas:
            Y = torch.clone(Y0).to(device=self.device, dtype=torch.float64)
            Y[cum_sums[noise_begin]:] += torch.normal(torch.zeros(
                (N_STATES - cum_sums[noise_begin],), dtype=torch.float64, device=self.device), std=sigma)
            finish = max_iter
            curr_corr = []
            if verbose:
                for i in range(len(cum_sums) - 1):
                    print(i + 1, torch.norm(Y[cum_sums[i]:cum_sums[i+1]
                                              ] - self.distances[cum_sums[i]:cum_sums[i+1]]))
            for j in range(max_iter):
                Y = (1 - alpha) * Y + alpha * \
                    (Y[self.neighbors_tensor].min(dim=1)[0] + 1)
                Y[0] = 0
                if verbose:
                    for i in range(len(cum_sums) - 1):
                        print(
                            i + 1, torch.norm(Y[cum_sums[i]:cum_sums[i+1]] - self.distances[cum_sums[i]:cum_sums[i+1]]))
                diff = torch.norm(Y - self.distances)
                curr_corr.append(correlation_func(
                    Y.cpu(), self.distances.cpu()).statistic)
                if verbose:
                    print(diff)
                if diff < eps:
                    finish = j
                    break
            iterations.append(finish + 1)
            correlations.append(curr_corr)
        print(round(alpha, 2), finish, torch.norm(
            Y - self.distances).cpu().numpy())
        return iterations, correlations
