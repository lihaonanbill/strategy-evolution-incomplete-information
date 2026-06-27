"""Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
from .generate import *
from .graph_operation import *
from itertools import product
import random
import math

def enumerate_profiles(num_strategies):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    profiles = list(product(*(range(1, n + 1) for n in num_strategies)))
    return profiles

def get_subgame_payoff(subgame_payoffs, subgame_profile, subgame_players, num_strategies):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    idx = 0
    base = 1
    for s, p in zip(reversed(subgame_profile), reversed(subgame_players)):
        player_idx = int(p) - 1
        idx += (s - 1) * base
        base *= num_strategies[player_idx]
    return subgame_payoffs[idx]

def calculate_global_payoffs(profile, games, payoffs, num_strategies):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    player_payoffs = [0 for _ in range(len(num_strategies))]
    'Compute payoffs, strategy updates, and transition matrices for networked evolutionary games.'
    for game_name, players in games.items():
        subgame_profile = tuple((int(profile[int(p) - 1]) for p in players))
        subgame_payoffs = payoffs[game_name]
        payoff_for_players = get_subgame_payoff(subgame_payoffs, subgame_profile, players, num_strategies)
        'Compute payoffs, strategy updates, and transition matrices for networked evolutionary games.'
        for i, p in enumerate(players):
            player_payoffs[int(p) - 1] += payoff_for_players[i]
    return player_payoffs

def caluculate_global_payoffs_iei(profile, games, payoffs, num_strategies, san, return_full_info=False):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    n_players = len(num_strategies)
    player_payoffs = [0 for _ in range(n_players)]
    accessible_sets = {}
    for i in range(1, n_players + 1):
        i_str = str(i)
        neigh = san.get(i_str, [])
        neigh_str = [str(x) for x in neigh]
        accessible_sets[i_str] = set(neigh_str) | {i_str}
    has_any_visible = [False] * n_players
    for game_name, players in games.items():
        players_str = [str(p) for p in players]
        players_set = set(players_str)
        subgame_profile = tuple((int(profile[int(p) - 1]) for p in players_str))
        subgame_payoffs = payoffs[game_name]
        payoff_for_players = get_subgame_payoff(subgame_payoffs, subgame_profile, players_str, num_strategies)
        for local_idx, p in enumerate(players_str):
            p_idx = int(p) - 1
            visible_to_p = players_set.issubset(accessible_sets.get(p, {p}))
            if visible_to_p:
                has_any_visible[p_idx] = True
                player_payoffs[p_idx] += payoff_for_players[local_idx]
    can_update = has_any_visible
    return (player_payoffs, can_update) if return_full_info else player_payoffs

def find_next_best_profile(profile, games, payoffs, num_strategies, san):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    'Compute payoffs, strategy updates, and transition matrices for networked evolutionary games.'
    current_payoffs, can_update = caluculate_global_payoffs_iei(profile, games, payoffs, num_strategies, san, return_full_info=True)
    new_profile = list(profile)
    for player in range(len(num_strategies)):
        if not can_update[player]:
            new_profile[player] = profile[player]
            continue
        best_strategy = profile[player]
        best_payoff = current_payoffs[player]
        for s in range(1, num_strategies[player] + 1):
            if s == profile[player]:
                continue
            test_profile = list(profile)
            test_profile[player] = s
            test_payoffs = caluculate_global_payoffs_iei(test_profile, games, payoffs, num_strategies, san)
            if test_payoffs[player] > best_payoff:
                best_payoff = test_payoffs[player]
                best_strategy = s
            elif test_payoffs[player] == best_payoff:
                best_strategy = max(best_strategy, s)
        new_profile[player] = best_strategy
    return tuple(new_profile)

def unconditional_imitate(profile, games, payoffs, num_strategies, san):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    n_players = len(num_strategies)
    global_payoffs = calculate_global_payoffs(profile, games, payoffs, num_strategies)
    accessible_sets = {}
    for i in range(1, n_players + 1):
        i_str = str(i)
        neigh = san.get(i_str, [])
        neigh_str = [str(x) for x in neigh]
        accessible_sets[i_str] = set(neigh_str) | {i_str}
    new_profile = list(profile)
    for i in range(n_players):
        i_str = str(i + 1)
        candidates = accessible_sets.get(i_str, {i_str})
        best_player = i + 1
        best_payoff = global_payoffs[i]
        best_strategy = profile[i]
        for p in candidates:
            try:
                p_idx = int(p) - 1
            except ValueError:
                continue
            if p_idx < 0 or p_idx >= n_players:
                continue
            cand_strategy = profile[p_idx]
            if cand_strategy < 1 or cand_strategy > num_strategies[i]:
                continue
            cand_payoff = global_payoffs[p_idx]
            cand_player = p_idx + 1
            if cand_payoff > best_payoff:
                best_payoff = cand_payoff
                best_strategy = cand_strategy
                best_player = cand_player
            elif cand_payoff == best_payoff:
                if cand_strategy > best_strategy:
                    best_strategy = cand_strategy
                    best_player = cand_player
                elif cand_strategy == best_strategy:
                    if cand_player > best_player:
                        best_player = cand_player
        new_profile[i] = best_strategy
    return tuple(new_profile)

def unconditional_imitate_normalized(profile, games, payoffs, num_strategies, san):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    n_players = len(num_strategies)
    global_payoffs = calculate_global_payoffs(profile, games, payoffs, num_strategies)
    deg = [0] * n_players
    for _, players in games.items():
        for p in players:
            p_idx = int(str(p)) - 1
            if 0 <= p_idx < n_players:
                deg[p_idx] += 1
    norm_payoffs = [global_payoffs[i] / (deg[i] if deg[i] > 0 else 1) for i in range(n_players)]
    accessible_sets = {}
    for i in range(1, n_players + 1):
        i_str = str(i)
        neigh = san.get(i_str, [])
        neigh_str = [str(x) for x in neigh]
        accessible_sets[i_str] = set(neigh_str) | {i_str}
    new_profile = list(profile)
    for i in range(n_players):
        i_str = str(i + 1)
        candidates = accessible_sets.get(i_str, {i_str})
        best_player = i + 1
        best_payoff = norm_payoffs[i]
        best_strategy = profile[i]
        for p in candidates:
            try:
                p_idx = int(p) - 1
            except ValueError:
                continue
            if p_idx < 0 or p_idx >= n_players:
                continue
            cand_strategy = profile[p_idx]
            if cand_strategy < 1 or cand_strategy > num_strategies[i]:
                continue
            cand_payoff = norm_payoffs[p_idx]
            cand_player = p_idx + 1
            if cand_payoff > best_payoff:
                best_payoff = cand_payoff
                best_strategy = cand_strategy
                best_player = cand_player
            elif cand_payoff == best_payoff:
                if cand_strategy > best_strategy:
                    best_strategy = cand_strategy
                    best_player = cand_player
                elif cand_strategy == best_strategy:
                    if cand_player > best_player:
                        best_player = cand_player
        new_profile[i] = best_strategy
    return tuple(new_profile)

def get_transition_matrix(games, payoffs, num_strategies, san):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    all_profiles = enumerate_profiles(num_strategies)
    profile_to_index = {profile: idx + 1 for idx, profile in enumerate(all_profiles)}
    next_profile_list = []
    for profile in all_profiles:
        next_profile = find_next_best_profile(profile, games, payoffs, num_strategies, san)
        index_of_next = profile_to_index[next_profile]
        next_profile_list.append(index_of_next)
    return next_profile_list

def get_transition_matrix_policy(games, payoffs, num_strategies, san, update_rule=1):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    all_profiles = enumerate_profiles(num_strategies)
    profile_to_index = {profile: idx + 1 for idx, profile in enumerate(all_profiles)}
    next_profile_list = []
    for profile in all_profiles:
        if update_rule == 1:
            next_profile = find_next_best_profile(profile, games, payoffs, num_strategies, san)
        elif update_rule == 2:
            next_profile = unconditional_imitate(profile, games, payoffs, num_strategies, san)
        elif update_rule == 3:
            next_profile = unconditional_imitate_normalized(profile, games, payoffs, num_strategies, san)
        else:
            raise ValueError('update_rule must be 1 (MBRA), 2 (UI), or 3 (UI normalized)')
        next_profile_list.append(profile_to_index[next_profile])
    return next_profile_list

def san_generate(games, rate, seed=None):
    """Compute payoffs, strategy updates, and transition matrices for networked evolutionary games."""
    if not 0 < rate <= 1:
        raise ValueError('rate must be in (0, 1].')
    rng = random.Random(seed)
    all_players = set()
    for _, players in games.items():
        for p in players:
            all_players.add(str(p))
    neighbor = {p: set() for p in all_players}
    for _, players in games.items():
        players_str = [str(p) for p in players]
        for p in players_str:
            neighbor[p].update((q for q in players_str if q != p))
    san = {}
    for p in sorted(all_players, key=lambda x: int(x) if x.isdigit() else x):
        neigh_list = sorted(neighbor[p], key=lambda x: int(x) if x.isdigit() else x)
        deg = len(neigh_list)
        if deg == 0:
            san[p] = []
            continue
        nan = int(math.floor(rate * deg))
        nan = max(1, min(nan, deg))
        san[p] = rng.sample(neigh_list, k=nan)
    return san
