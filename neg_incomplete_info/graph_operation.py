from itertools import product
import numpy as np
from collections import deque
from typing import List, Sequence
from collections import Counter
from typing import Sequence, List
from typing import List, Dict, Tuple

def enumerate_profiles(num_strategies):
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    return list(product(*(range(1, n + 1) for n in num_strategies)))

def draw_directed_graph(edge_lists, num_strategies):
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    import networkx as nx
    import matplotlib.pyplot as plt

    profiles = enumerate_profiles(num_strategies)
    n_nodes = len(profiles)
    G = nx.DiGraph()
    for i, prof in enumerate(profiles, start=1):
        label = str(prof)
        G.add_node(i, label=label)
    for edge_list in edge_lists:
        for i, target in enumerate(edge_list):
            if 1 <= target <= n_nodes:
                G.add_edge(i + 1, target)
    pos = nx.shell_layout(G)
    nx.draw(G, pos, with_labels=True, labels={i: G.nodes[i]['label'] for i in G.nodes}, node_size=1200, node_color='lightblue', arrowsize=20, font_size=10)
    plt.title('Message omitted from public release.')
    plt.show()

def find_all_cycles(edge_lists, num_strategies):
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    if not edge_lists:
        print('Message omitted from public release.')
        return []
    profiles = enumerate_profiles(num_strategies)
    n = len(profiles)
    combined_graph = [set() for _ in range(n)]
    for edge_list in edge_lists:
        if len(edge_list) != n:
            raise ValueError('edge_list length does not match number of states implied by num_strategies.')
        for i, target in enumerate(edge_list):
            combined_graph[i].add(target)
    graph = [list(targets) for targets in combined_graph]
    visited = [0] * n
    path = []
    cycles = []
    seen_cycles = set()

    def canonical_cycle_key(cycle_states_1based):
        """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
        cyc = cycle_states_1based
        k = len(cyc)
        if k == 0:
            return tuple()
        mins = min(cyc)
        candidates = []
        for i in range(k):
            if cyc[i] == mins:
                candidates.append(tuple(cyc[i:] + cyc[:i]))
        return min(candidates)

    def dfs(node):
        visited[node] = 1
        path.append(node + 1)
        for next_node in graph[node]:
            next_idx = next_node - 1
            if visited[next_idx] == 0:
                dfs(next_idx)
            elif visited[next_idx] == 1:
                cycle_start = path.index(next_node)
                cycle_states = path[cycle_start:]
                key = canonical_cycle_key(cycle_states)
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycle_profiles = [profiles[s - 1] for s in cycle_states]
                    cycles.append({'states': cycle_states, 'profiles': cycle_profiles})
        visited[node] = 2
        path.pop()
    for i in range(n):
        if visited[i] == 0:
            dfs(i)
    rev = [[] for _ in range(n)]
    for u in range(n):
        for v_1based in graph[u]:
            v = v_1based - 1
            rev[v].append(u)

    def basin_size_for_cycle(cycle_states_1based):
        """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
        stack = [s - 1 for s in cycle_states_1based]
        seen = set(stack)
        while stack:
            v = stack.pop()
            for pre in rev[v]:
                if pre not in seen:
                    seen.add(pre)
                    stack.append(pre)
        return len(seen)
    for c in cycles:
        c['basin_size'] = basin_size_for_cycle(c['states'])
    if not cycles:
        print('Message omitted from public release.')
    else:
        print('Message omitted from public release.')
        for c in cycles:
            print('states:', c['states'])
            print('profiles:', c['profiles'])
            print('basin_size:', c['basin_size'])
            print('-' * 40)
    return cycles

def _to_zero_based(transition):
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    nxt = np.asarray(transition, dtype=int)
    S = len(nxt)
    mn, mx = (int(nxt.min()), int(nxt.max()))
    if mn >= 1 and mx <= S:
        return ((nxt - 1).tolist(), True)
    if mn >= 0 and mx < S:
        return (nxt.tolist(), False)
    raise ValueError(f'Transition values out of range. min={mn}, max={mx}, expected 0..{S - 1} or 1..{S}.')
    'Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics.'
    next0, one_based = _to_zero_based(transition)
    S = len(next0)
    state = [0] * S
    stack = []
    pos_in_stack = {}
    cycles = []
    attractor_id = [-1] * S

    def dfs(start):
        v = start
        while True:
            if state[v] == 0:
                state[v] = 1
                pos_in_stack[v] = len(stack)
                stack.append(v)
                v = next0[v]
                continue
            if state[v] == 1:
                cycle_start_idx = pos_in_stack[v]
                cyc = stack[cycle_start_idx:]
                cid = len(cycles)
                cycles.append(cyc)
                for u in cyc:
                    attractor_id[u] = cid
                break
            break
        for u in stack:
            state[u] = 2
        stack.clear()
        pos_in_stack.clear()
    for i in range(S):
        if state[i] == 0:
            dfs(i)
    rev = [[] for _ in range(S)]
    for i, j in enumerate(next0):
        rev[j].append(i)
    distance = [-1] * S
    q = deque()
    for cid, cyc in enumerate(cycles):
        for u in cyc:
            distance[u] = 0
            q.append(u)
    while q:
        u = q.popleft()
        for pred in rev[u]:
            if distance[pred] != -1:
                continue
            distance[pred] = distance[u] + 1
            attractor_id[pred] = attractor_id[u]
            q.append(pred)
    return (cycles, attractor_id, distance, next0, one_based)

def analyze_functional_graph(transition, *, _assume_zero_based: bool=False):
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    if _assume_zero_based:
        next0 = list(map(int, transition))
        one_based = False
        S = len(next0)
        if S > 0 and (min(next0) < 0 or max(next0) >= S):
            raise ValueError(f'next_node values out of range. expected 0..{S - 1}.')
    else:
        next0, one_based = _to_zero_based(transition)
        S = len(next0)
    state = [0] * S
    stack = []
    pos_in_stack = {}
    cycles = []
    attractor_id = [-1] * S

    def walk_from(start):
        """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
        v = start
        while True:
            if state[v] == 0:
                state[v] = 1
                pos_in_stack[v] = len(stack)
                stack.append(v)
                v = next0[v]
                continue
            if state[v] == 1:
                cycle_start_idx = pos_in_stack[v]
                cyc = stack[cycle_start_idx:]
                cid = len(cycles)
                cycles.append(cyc)
                for u in cyc:
                    attractor_id[u] = cid
                break
            break
        for u in stack:
            state[u] = 2
        stack.clear()
        pos_in_stack.clear()
    for i in range(S):
        if state[i] == 0:
            walk_from(i)
    rev = [[] for _ in range(S)]
    for i, j in enumerate(next0):
        rev[j].append(i)
    distance = [-1] * S
    q = deque()
    for cid, cyc in enumerate(cycles):
        for u in cyc:
            distance[u] = 0
            q.append(u)
    while q:
        u = q.popleft()
        for pred in rev[u]:
            if distance[pred] != -1:
                continue
            distance[pred] = distance[u] + 1
            attractor_id[pred] = attractor_id[u]
            q.append(pred)
    return (cycles, attractor_id, distance, next0, one_based)

def plot_all_state_trajectories(transition, T=None, figsize=(12, 5), linewidth=0.6, alpha=0.5, *, show=True, close=False):
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    import matplotlib.pyplot as plt

    cycles, attractor_id, dist, next0, one_based = analyze_functional_graph(transition)
    S = len(next0)
    max_dist = max(dist) if dist else 0
    max_cyc = max((len(c) for c in cycles)) if cycles else 1
    if T is None:
        T = max_dist + max_cyc + 2
    traj = np.zeros((S, T + 1), dtype=int)
    for s0 in range(S):
        x = s0
        traj[s0, 0] = x
        for t in range(T):
            x = next0[x]
            traj[s0, t + 1] = x
    fig, ax = plt.subplots(figsize=figsize)
    ts = np.arange(T + 1)
    offset = 1 if one_based else 0
    for s0 in range(S):
        ax.plot(ts, traj[s0] + offset, linewidth=linewidth, alpha=alpha)
    ax.set_title(f'Trajectories (S={S}, T={T})')
    ax.set_xlabel('Time t')
    ax.set_ylabel('State')
    ax.set_ylim(offset - 0.5, S - 1 + offset + 0.5)
    fig.tight_layout()
    if show:
        plt.show()
        if close:
            plt.close(fig)
    return (fig, ax)

def plot_state_transition_graph(f_1based, figsize=(10, 7), seed=0, show_labels=False, savepath=None, arrows_threshold=500, basin_R=10.0, cycle_radius=0.6, layer_gap=0.55, max_per_ring=80, cycle_jitter=0.03, transient_jitter=0.05, edge_alpha=0.35, edge_lw=0.5, node_alpha=0.95, transient_size=8, attractor_size=70):
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    rng = np.random.default_rng(seed)
    n = len(f_1based)
    nxt = np.array(f_1based, dtype=int) - 1
    if np.any(nxt < 0) or np.any(nxt >= n):
        raise ValueError('f_1based must contain integers in [1, n].')
    cycles, attractor_id, distance, _next0, _one_based = analyze_functional_graph(nxt.tolist(), _assume_zero_based=True)
    dest_cycle = np.array(attractor_id, dtype=int)
    dist_to_cycle = np.array(distance, dtype=int)
    is_attractor = dist_to_cycle == 0
    cycle_id = np.full(n, -1, dtype=int)
    for cid, cyc in enumerate(cycles):
        for u in cyc:
            cycle_id[u] = cid
    basins = {cid: {} for cid in range(len(cycles))}
    for u in range(n):
        cid = dest_cycle[u] if dest_cycle[u] != -1 else 0
        d = dist_to_cycle[u] if dist_to_cycle[u] != -1 else 0
        basins.setdefault(cid, {}).setdefault(d, []).append(u)
    m = max(1, len(cycles))
    if m == 1:
        centers = {0: np.array([0.0, 0.0])}
    else:
        centers = {}
        for cid in range(m):
            ang = 2 * np.pi * cid / m
            centers[cid] = np.array([basin_R * np.cos(ang), basin_R * np.sin(ang)])
    pos = np.zeros((n, 2), dtype=float)
    for cid, cyc in enumerate(cycles if cycles else [[0]]):
        c0 = centers.get(cid, np.array([0.0, 0.0]))
        k = len(cyc)
        if k == 1:
            pos[cyc[0]] = c0
        else:
            for idx, u in enumerate(cyc):
                ang = 2 * np.pi * idx / k
                jitter = rng.normal(scale=cycle_jitter, size=2)
                pos[u] = c0 + cycle_radius * np.array([np.cos(ang), np.sin(ang)]) + jitter
        layers = basins.get(cid, {})
        for d, nodes in layers.items():
            if d == 0:
                continue
            nodes = nodes.copy()
            rng.shuffle(nodes)
            rings = int(np.ceil(len(nodes) / max_per_ring))
            for r in range(rings):
                chunk = nodes[r * max_per_ring:(r + 1) * max_per_ring]
                if not chunk:
                    continue
                radius = cycle_radius + layer_gap * (d + 0.7 * r)
                cnt = len(chunk)
                rot = rng.uniform(0, 2 * np.pi)
                for j, u in enumerate(chunk):
                    ang = rot + 2 * np.pi * j / cnt
                    jitter = rng.normal(scale=transient_jitter, size=2)
                    pos[u] = c0 + radius * np.array([np.cos(ang), np.sin(ang)]) + jitter
    if len(cycles) == 0:
        pos = rng.normal(size=(n, 2))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')
    edges = [(u, int(nxt[u])) for u in range(n)]
    if n <= arrows_threshold:
        for u, v in edges:
            p1, p2 = (pos[u], pos[v])
            vec = p2 - p1
            L = np.linalg.norm(vec) + 1e-09
            shrink = 0.08
            q1 = p1 + shrink * vec / L
            q2 = p2 - shrink * vec / L
            arrow = FancyArrowPatch(q1, q2, arrowstyle='-|>', mutation_scale=6, linewidth=edge_lw, color='0.65', alpha=edge_alpha)
            ax.add_patch(arrow)
    else:
        segs = [(pos[u], pos[v]) for u, v in edges]
        lc = LineCollection(segs, colors='0.7', linewidths=edge_lw, alpha=edge_alpha)
        ax.add_collection(lc)
    transient = np.where(~is_attractor)[0]
    attractor = np.where(is_attractor)[0]
    ax.scatter(pos[transient, 0], pos[transient, 1], s=transient_size, c='#1f77b4', alpha=node_alpha, linewidths=0)
    ax.scatter(pos[attractor, 0], pos[attractor, 1], s=attractor_size, c='#ff7f0e', alpha=node_alpha, linewidths=0)
    if show_labels and n <= 80:
        for u in range(n):
            ax.text(pos[u, 0], pos[u, 1], str(u + 1), fontsize=8, ha='center', va='center', color='black' if is_attractor[u] else 'white')
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

def mean_convergent_steps(transition: Sequence[int], *, assume_zero_based: bool=False) -> float:
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    cycles, attractor_id, distance, next0, one_based = analyze_functional_graph(transition, _assume_zero_based=assume_zero_based)
    N = len(distance)
    if N == 0:
        return 0.0
    if any((d < 0 for d in distance)):
        raise ValueError('distance contains negative value. Input mapping may be invalid or disconnected.')
    return sum(distance) / N

def expected_cooperation_percentage_calculation(transition: Sequence[int], num_strategies: List[int]) -> float:
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    'Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics.'
    cycles, attractor_id, _distance, _next0, _one_based = analyze_functional_graph(transition)
    S = len(transition)
    n_players = len(num_strategies)
    if S == 0:
        return 0.0
    if n_players == 0:
        raise ValueError('num_strategies must be non-empty.')

    def decode_profile(state_id0: int) -> List[int]:
        x = state_id0
        profile = [1] * n_players
        for i in range(n_players - 1, -1, -1):
            base = int(num_strategies[i])
            if base <= 0:
                raise ValueError('Each num_strategies[i] must be positive.')
            digit0 = x % base
            profile[i] = digit0 + 1
            x //= base
        return profile
    basin_counts = Counter(attractor_id)
    if -1 in basin_counts:
        raise RuntimeError('Some states have attractor_id = -1; functional graph analysis may be incomplete.')
    Ci = [0.0] * len(cycles)
    for cid, cyc in enumerate(cycles):
        if not cyc:
            Ci[cid] = 0.0
            continue
        coop_rates = []
        for state0 in cyc:
            prof = decode_profile(state0)
            coop = sum((1 for a in prof if a == 1))
            coop_rates.append(coop / n_players)
        Ci[cid] = sum(coop_rates) / len(coop_rates)
    expected = 0.0
    for cid in range(len(cycles)):
        Pi = basin_counts.get(cid, 0) / S
        expected += Pi * Ci[cid]
    return expected

def deviate_score(baseline_transition: List[int], variant_transition: List[int], *, return_details: bool=False) -> float | Tuple[float, Dict[str, object]]:
    """Analyze functional graphs, attractors, basins, cooperation rates, and convergence metrics."""
    if len(baseline_transition) != len(variant_transition):
        raise ValueError('Message omitted from public release.')
    base_cycles, base_attr_id, _, _, _ = analyze_functional_graph(baseline_transition)
    var_cycles, var_attr_id, _, _, _ = analyze_functional_graph(variant_transition)
    N = len(base_attr_id)
    base_cycle_sets = [frozenset(cyc) for cyc in base_cycles]
    var_cycle_sets = [frozenset(cyc) for cyc in var_cycles]
    unchanged = 0
    for i in range(N):
        base_cycle = base_cycle_sets[base_attr_id[i]]
        var_cycle = var_cycle_sets[var_attr_id[i]]
        if base_cycle == var_cycle:
            unchanged += 1
    similarity = unchanged / N
    dev = 1.0 - similarity
    if not return_details:
        return dev
    details = {'N': N, 'unchanged_states': unchanged, 'similarity_score': similarity, 'deviate_score': dev, 'baseline_num_attractors': len(base_cycles), 'variant_num_attractors': len(var_cycles), 'baseline_cycles': [tuple(sorted(c)) for c in base_cycle_sets], 'variant_cycles': [tuple(sorted(c)) for c in var_cycle_sets]}
    return (dev, details)
if __name__ == '__main__':
    f = [3, 2, 4, 1]
    plot_state_transition_graph(f, show_labels=True, basin_R=12, layer_gap=0.7, edge_alpha=0.25)
