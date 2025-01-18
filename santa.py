import pandas as pd
import json
from sympy.combinatorics.permutations import Permutation
import matplotlib.pyplot as plt
import numpy as np

def get_moves(puzzle_type: str, filepath: str) -> dict[str, list[int]]:
    """
    Generates moves in the format `move name` -> `permutation of states`.

    Args:
        puzzle_type: A string description of the puzzle (e.g., "cube_2/2/2", "cube_3/3/3", etc)
        filepath: Path to the `puzzle_info.csv` file.

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
