"""Generate interaction structures and public-goods-game payoff tables."""
import random
from typing import Dict, List, Tuple, Set, Optional, Literal, Union
from numbers import Number

def generate_games(n_players: int, graph_type: Literal['pairwise', 'hypergraph', 'mixed'], structure_spec: Optional[dict]=None, seed: Optional[int]=None) -> Dict[str, List[str]]:
    """Generate interaction structures and public-goods-game payoff tables."""
    if structure_spec is None:
        structure_spec = {}
    if seed is None:
        seed = structure_spec.get('seed', None)
    rng = random.Random(seed)
    allow_approx = bool(structure_spec.get('allow_approx', True))
    max_trials = int(structure_spec.get('max_trials', 2000))
    if n_players <= 0:
        raise ValueError('n_players must be positive.')
    nodes: List[int] = list(range(1, n_players + 1))

    def _neighbors_from_edges2(edges2: Set[Tuple[int, int]]) -> Dict[int, Set[int]]:
        nb = {i: set() for i in nodes}
        for a, b in edges2:
            nb[a].add(b)
            nb[b].add(a)
        return nb

    def _sample_nodes_with_positive_need(need: List[int], k: int) -> List[int]:
        """Generate interaction structures and public-goods-game payoff tables."""
        candidates = [idx + 1 for idx, v in enumerate(need) if v > 0]
        if len(candidates) < k:
            raise ValueError('Not enough nodes with remaining degree/participation needs.')
        return rng.sample(candidates, k)

    def _build_pairwise_target_degrees(d2: int) -> List[int]:
        if d2 < 0:
            raise ValueError('d2 must be >= 0.')
        if d2 > n_players - 1:
            raise ValueError(f'd2={d2} is too large for N={n_players} (need d2 <= N-1).')
        deg = [d2] * n_players
        total = n_players * d2
        if total % 2 != 0:
            if not allow_approx:
                raise ValueError('N*d2 is odd; impossible for simple undirected graph with all degrees=d2.')
            if d2 == 0:
                raise ValueError('N*d2 is odd but d2=0; cannot apply approx (-1) fix.')
            v = rng.choice(nodes)
            deg[v - 1] -= 1
        return deg

    def _build_hyper_target_participations(d3: int) -> List[int]:
        if d3 < 0:
            raise ValueError('d3 must be >= 0.')
        need = [d3] * n_players
        total = n_players * d3
        r = total % 3
        if r != 0:
            if not allow_approx:
                raise ValueError('N*d3 is not divisible by 3; impossible for 3-uniform hypergraph with all participations=d3.')
            if d3 == 0:
                raise ValueError('N*d3 % 3 != 0 but d3=0; cannot apply approx (-1) fix.')
            k = r
            chosen = rng.sample(nodes, k)
            for v in chosen:
                need[v - 1] -= 1
        return need

    def _generate_pairwise_edges(target_deg: List[int]) -> Set[Tuple[int, int]]:
        """Generate interaction structures and public-goods-game payoff tables."""
        for _attempt in range(max_trials):
            remaining = target_deg[:]
            edges2: Set[Tuple[int, int]] = set()
            ok = True
            while True:
                pos = [i for i, d in enumerate(remaining) if d > 0]
                if not pos:
                    break
                maxd = max((remaining[i] for i in pos))
                candidates = [i for i in pos if remaining[i] == maxd]
                a = rng.choice(candidates) + 1
                possible_b = []
                for j in pos:
                    b = j + 1
                    if b == a:
                        continue
                    e = (a, b) if a < b else (b, a)
                    if e in edges2:
                        continue
                    possible_b.append(b)
                if not possible_b:
                    ok = False
                    break
                b = rng.choice(possible_b)
                e = (a, b) if a < b else (b, a)
                edges2.add(e)
                remaining[a - 1] -= 1
                remaining[b - 1] -= 1
                if remaining[a - 1] < 0 or remaining[b - 1] < 0:
                    ok = False
                    break
            if ok:
                return edges2
        raise RuntimeError('Failed to generate pairwise edges satisfying target degrees; try larger N or smaller d2/max_trials.')

    def _generate_hyperedges3(target_need: List[int]) -> Set[Tuple[int, int, int]]:
        """Generate interaction structures and public-goods-game payoff tables."""
        for _attempt in range(max_trials):
            remaining = target_need[:]
            edges3: Set[Tuple[int, int, int]] = set()
            ok = True
            while True:
                pos = [i for i, d in enumerate(remaining) if d > 0]
                if not pos:
                    break
                maxd = max((remaining[i] for i in pos))
                candidates = [i for i in pos if remaining[i] == maxd]
                a = rng.choice(candidates) + 1
                found = False
                for _ in range(200):
                    others = [j + 1 for j in pos if j + 1 != a]
                    if len(others) < 2:
                        found = False
                        break
                    b, c = rng.sample(others, 2)
                    tri = tuple(sorted((a, b, c)))
                    if tri in edges3:
                        continue
                    found = True
                    break
                if not found:
                    ok = False
                    break
                edges3.add(tri)
                for v in tri:
                    remaining[v - 1] -= 1
                    if remaining[v - 1] < 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return edges3
        raise RuntimeError('Failed to generate 3-hyperedges satisfying target participations; try larger N or smaller d3/max_trials.')
    edges2: Set[Tuple[int, int]] = set()
    edges3: Set[Tuple[int, int, int]] = set()
    if graph_type == 'pairwise':
        d2 = int(structure_spec.get('d2'))
        target_deg2 = _build_pairwise_target_degrees(d2)
        edges2 = _generate_pairwise_edges(target_deg2)
    elif graph_type == 'hypergraph':
        d3 = int(structure_spec.get('d3'))
        target_need3 = _build_hyper_target_participations(d3)
        edges3 = _generate_hyperedges3(target_need3)
    elif graph_type == 'mixed':
        d2 = int(structure_spec.get('d2'))
        d3 = int(structure_spec.get('d3'))
        target_deg2 = _build_pairwise_target_degrees(d2)
        target_need3 = _build_hyper_target_participations(d3)
        edges2 = _generate_pairwise_edges(target_deg2)
        edges3 = _generate_hyperedges3(target_need3)
    else:
        raise ValueError(f'Unknown graph_type: {graph_type}')
    games: Dict[str, List[str]] = {}
    idx = 1
    for a, b in sorted(edges2):
        games[f'game{idx}'] = [str(a), str(b)]
        idx += 1
    for a, b, c in sorted(edges3):
        games[f'game{idx}'] = [str(a), str(b), str(c)]
        idx += 1
    return games

def generate_payoffs(games: Dict[str, List[str]], *, r_over_n: Number) -> Dict[str, List[List[Number]]]:
    """Generate interaction structures and public-goods-game payoff tables."""
    if r_over_n is None:
        raise ValueError('r_over_n must be provided.')
    if r_over_n < 0:
        raise ValueError('r_over_n must be non-negative.')
    c: Number = 1.0
    payoffs: Dict[str, List[List[Number]]] = {}
    for game_name, players in games.items():
        k = len(players)
        if k < 2:
            raise ValueError(f'games[{game_name}] has size {k}. Require k>=2.')
        r = r_over_n * k
        sub: List[List[Number]] = []

        def backtrack(idx: int, profile: List[int]) -> None:
            if idx == k:
                kC = sum((1 for s in profile if s == 1))
                total_contrib = kC * c
                share = r * total_contrib / k
                payoff_vec = [share - c if s == 1 else share for s in profile]
                sub.append(payoff_vec)
                return
            for s in (1, 2):
                profile.append(s)
                backtrack(idx + 1, profile)
                profile.pop()
        backtrack(0, [])
        if len(sub) != 1 << k or any((len(x) != k for x in sub)):
            raise RuntimeError(f'Internal error: payoff shape mismatch for k={k}.')
        payoffs[game_name] = sub
    return payoffs
