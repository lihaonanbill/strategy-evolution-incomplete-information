"""Run a small reproducibility demo without external data."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neg_incomplete_info.graph_operation import (
    deviate_score,
    expected_cooperation_percentage_calculation,
    mean_convergent_steps,
)
from neg_incomplete_info.transition_matrix import (
    generate_games,
    generate_payoffs,
    get_transition_matrix_policy,
    san_generate,
)


def main() -> None:
    n_players = 6
    num_strategies = [2] * n_players

    games = generate_games(
        n_players=n_players,
        graph_type="mixed",
        structure_spec={"d2": 2, "d3": 1, "allow_approx": True},
        seed=42,
    )
    payoffs = generate_payoffs(games, r_over_n=0.75)

    san_partial = san_generate(games, rate=0.5, seed=1)
    san_full = san_generate(games, rate=1.0, seed=1)

    transition_partial = get_transition_matrix_policy(
        games, payoffs, num_strategies, san_partial, update_rule=2
    )
    transition_full = get_transition_matrix_policy(
        games, payoffs, num_strategies, san_full, update_rule=2
    )

    cooperation = expected_cooperation_percentage_calculation(
        transition_partial, num_strategies
    )
    deviation = deviate_score(transition_full, transition_partial)
    steps = mean_convergent_steps(transition_partial)

    print("Demo network games:", len(games))
    print("States:", len(transition_partial))
    print(f"Cooperation percentage: {cooperation:.4f}")
    print(f"Deviate score: {deviation:.4f}")
    print(f"Mean convergent steps: {steps:.4f}")


if __name__ == "__main__":
    main()
