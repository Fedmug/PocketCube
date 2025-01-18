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
from santa import get_moves


# God's number (in QTM) for Pocket Cube
N_GOD = 14

# The number of all states for Pocket Cube
N_STATES = 3674160


def row_difference(A, B: torch.tensor) -> torch.tensor:
    r"""
        Calulates A \ B — set difference between rows of A and rows of B
    """
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    return A[~mask[:len(A)]]


def rowwise_pearson_correlation(A, b: torch.Tensor, eps: float = 1e-10):
    assert len(A.shape) == 2, "A must be a 2D tensor"
    assert len(b.shape) == 1, "b must be a 1D tensor"
    assert A.shape[1] == b.shape[0], "The number of columns in A must match the size of b"

    A_mean = A.mean(dim=1, keepdim=True)
    A_std = A.std(dim=1, unbiased=False, keepdim=True)
    b_mean = b.mean()
    b_std = b.std(unbiased=False)

    A_norm = (A - A_mean) / (A_std + eps)
    b_norm = (b - b_mean) / (b_std + eps)

    return (A_norm @ b_norm) / b.shape[0]


class TensorCube:
    def __init__(self, device, puzzle_type: str = "cube_2/2/2",
                 generators: list[str] = [
                     'f0', '-f0', 'r0', '-r0', 'd0', '-d0'],
                 verbose: bool = True):
        self.device = device
        self.puzzle_type = puzzle_type
        self.generators = generators
        self.available_moves = self._moves2tensor()
        self.states = torch.zeros(
            N_STATES, self.state_size, device=self.device, dtype=torch.int8)
        self.distances = torch.zeros(
            N_STATES, dtype=torch.int8, device=self.device)
        self.layers_pop = np.zeros(N_GOD + 1, dtype=np.int32)
        self.visit_cube_states(verbose)
        self.neighbors_tensor = self._get_neighbors_tensor()
        self.init_names = (
            "all zeros",
            "1 true layer",
            "2 true layers",
            "3 true layers",
            "4 true layers",
            "5 true layers",
            "6 true layers",
            "7 true layers",
            "8 true layers",
            "9 true layers",
            "10 true layers",
            "11 true layers",
            "12 true layers",
            "13 true layers",
            "random walk",
            "random",
            "manhattan",
            "hamming",
            "catboost",
        )
        self.Y0 = self.init_distances()

    @property
    def n_gens(self):
        return len(self.generators)

    @property
    def state_size(self):
        return self.available_moves.shape[1]

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

    def make_random_walks_dataset(self,
                                  n_random_walk_length: int,
                                  n_random_walks_to_generate: int,
                                  rw_start=None,
                                  verbose: bool = False):
        '''    
        Args: 
            n_random_walk_length - number of visited nodes, i.e. number of steps + 1 
            n_random_walks_to_generate - how many random walks will run in parrallel
            rw_start - initial states for random walks - by default we will use 0,1,2,3 ...
                Can be vector or array
                If it is vector it will be broadcasted n_random_walks_to_generate times, 
                If it is array n_random_walks_to_generate - input n_random_walks_to_generate will be ignored
                and will be assigned: n_random_walks_to_generate = rw_start.shape[0]

        Returns:
            X, y: np.array, X - array of states, y - number of steps rw achieves it
        '''

        # Initialize current array of steps with  state_rw_start - expanding (broadcasting) it into array:
        if rw_start is None:
            # Starting state for random walks (for train data it is our destination state - "solved puzzle state")
            state_rw_start = torch.arange(self.state_size, device=self.device)
            array_of_states = state_rw_start.view(1, self.state_size).expand(
                n_random_walks_to_generate, self.state_size)
        else:
            if len(rw_start.shape) == 1:
                array_of_states = rw_start.view(1, self.state_size).expand(
                    n_random_walks_to_generate, self.state_size)
            else:
                array_of_states = rw_start
        if verbose:
            print('state_rw_start.shape:', state_rw_start.shape)
            print('state_rw_start:', state_rw_start)
            print(array_of_states.shape)
            print(array_of_states[:3, :])

        # Output: X,y - states, y - how many steps we achieve them
        # Allocate memory:
        X = torch.zeros(n_random_walks_to_generate *
                        n_random_walk_length, self.state_size, device=self.device)
        y = torch.zeros(n_random_walks_to_generate *
                        n_random_walk_length, device=self.device)
        if verbose:
            print('X.shape', X.shape)

        # First portion of data  - just our state_rw_start state  multiplexed many times
        X[:n_random_walks_to_generate, :] = array_of_states
        y[:n_random_walks_to_generate] = 0

        # Technical to make array[ IX_array] we need  actually to write array[ range(N), IX_array  ]
        row_indices = torch.arange(
            array_of_states.shape[0], device=self.device)
        row_indices = np.arange(array_of_states.shape[0])[:, np.newaxis]

        # Main loop
        for i_step in range(1, n_random_walk_length):
            y[(i_step)*n_random_walks_to_generate: (i_step+1)
              * n_random_walks_to_generate] = i_step
            IX_moves = np.random.randint(
                0, self.n_gens, size=n_random_walks_to_generate, dtype=int)  # random moves indixes
            # all_moves[IX_moves,:] ]
            new_array_of_states = array_of_states[row_indices,
                                                  self.available_moves[IX_moves, :]]
            array_of_states = new_array_of_states
            X[(i_step)*n_random_walks_to_generate: (i_step+1) *
              n_random_walks_to_generate, :] = new_array_of_states

        if verbose:
            print(array_of_states.shape, 'array_of_states.shape')
            print(n_random_walk_length, 'n_random_walk_length',
                  self.state_size, 'state_size', '')
            print('Finished')
            print(str(X)[:500])
            print(str(y)[:500])

        return X, y

    def cat_boost_init(self, X, y: np.array, verbose: bool = True):
        start_time = time()
        if verbose:
            print("Starting catboost init...")
        model = CatBoostRegressor(verbose=False)
        model.fit(X, y)
        if verbose:
            y_pred = model.predict(X)
            r2_train = r2_score(y, y_pred)
            print(f"Train R^2 score: {r2_train:.2f}")
        result = model.predict(self.states.cpu().numpy())
        if verbose:
            print(f"Catboost init took {time() - start_time:.2f} s")
        return torch.tensor(result, device=self.device)

    def init_distances(self):
        """
        Return variants of distance initializations:
        - all zeros
        - true distaces for first k layers, k=1,..,13
        - Manhattan distance to target
        - Hamming distance to target
        - random walks initialization
        - random value for each layer
        """
        RW_DIST = [1, 2, 3, 4, 5, 7, 10, 14, 18, 29, 65, 216, 385, 478]
        cum_sums = self.layers_pop.cumsum()
        Y = torch.zeros((len(self.init_names), N_STATES),
                        dtype=torch.float64, device=self.device)
        for k in range(1, 14):
            Y[k, :cum_sums[k]] = self.distances[:cum_sums[k]]
            Y[14, :cum_sums[k]] = RW_DIST[k]
            Y[15, :cum_sums[k]] = np.random.randint(15)
        norm = torch.linalg.vector_norm((self.states - self.states[0, :].unsqueeze(0)).to(
            device=self.device, dtype=torch.float64), 1, dim=-1)
        Y[16, :] = norm * 14. / norm.max()
        Y[17, :] = torch.cdist(self.states.to(device=self.device, dtype=torch.float64),
                               self.states[0].unsqueeze(0).to(
                                   device=self.device, dtype=torch.float64),
                               p=0).squeeze() * 14. / 24.
        X, y = self.make_random_walks_dataset(15, 100_000)
        Y[18, :] = self.cat_boost_init(X.cpu().numpy(), y.cpu().numpy())
        return Y

    def convergence_stats(self,
                          args: Iterable[int | float],
                          vs_alpha: bool = True,
                          eps: float = 1e-5,
                          max_iter: int = 1000,
                          sigma: float = 1.0,
                          verbose: bool = False):
        """
        Finds number of iterations and correlations on each iteration of DP algorithm

        Args:
            vs_alpha: bool, if true, collects stats vs alphas, otherwise vs noise begins
            args: list of alphas (floats from (0, 1]) or noise begins (range(N_GOD))
            eps: float, tolerance
            max_iter: int, maximum number of iterations
            sigma: float, std of gaussian noise
        """
        cum_sums = self.layers_pop.cumsum()
        iterations = torch.full(
            (self.Y0.shape[0], len(args)), max_iter, device=self.device)
        pearson = torch.full((self.Y0.shape[0], len(
            args), max_iter), 2.0, device=self.device, dtype=torch.float64)
        # spearman = torch.zeros((Y0.shape[0], len(alphas), max_iter))
        true_distances = self.distances.to(
            device=self.device, dtype=torch.float64)
        for i, arg in enumerate(args):
            iteration_begin = time()
            alpha = arg if vs_alpha else 1.0
            Y = torch.clone(self.Y0).to(
                device=self.device, dtype=torch.float64)
            if not vs_alpha:
                Y[:, cum_sums[i]:] += torch.normal(
                    torch.zeros((Y.shape[0], N_STATES - cum_sums[i],),
                                dtype=torch.float64, device=self.device),
                    std=sigma)

                Y = torch.clip(Y, 0, N_GOD)
            finish = max_iter
            curr_corr = []
            for j in range(max_iter):
                if verbose:
                    print(f"Starting iteration {j}...")
                start = time()
                Y = (1 - alpha) * Y + alpha * \
                    (Y[:, self.neighbors_tensor].min(dim=-1)[0] + 1)
                Y[:, 0] = 0
                if verbose:
                    print(f"Update took {time() - start:.2f} s")
                if verbose:
                    for i in range(len(cum_sums) - 1):
                        print(f"  layer {i + 1} error:",
                              torch.norm(Y[:, cum_sums[i]:cum_sums[i+1]] - true_distances[cum_sums[i]:cum_sums[i+1]], dim=-1).max())
                diff = torch.norm(Y - self.distances, dim=1)
                iterations[(iterations[:, i] == max_iter)
                           & (diff < eps), i] = j + 1
                pearson[:, i, j] = rowwise_pearson_correlation(
                    Y, true_distances)
                # spearman[:, i, j] = rowwise_spearman_correlation(Y, true_distances)
                # pearson_scipy = []
                # for idx in range(Y.shape[0]):
                #     pearson_scipy.append(pearsonr(Y[idx].cpu().numpy(), true_distances.cpu().numpy()).statistic)
                # corr_diff = np.linalg.norm(pearson_scipy - pearson[:, i, j].cpu().numpy())
                # if corr_diff > 1e-6:
                #     print("Too large diff:", corr_diff)
                # curr_corr.append(spearman(Y.T, torch.tile(self.distances.to(torch.float), (Y.shape[0], 1)).T))
                if verbose:
                    print(diff.cpu())
                if diff.max() < eps:
                    break
                if verbose:
                    print(f"iteration {j} took {time() - start:.2f} s")
            print(round(alpha, 2) if vs_alpha else f"noise from layer {i}",
                  torch.norm(Y - self.distances).cpu().numpy(),
                  f"took {time() - iteration_begin:.2f} s")
        return iterations.cpu().numpy(), pearson.cpu().numpy()
