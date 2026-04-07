import numpy as np

def generate_bilinear_symmetries():
    # 1. Generate all 945 perfect matchings for 10 legs
    def get_matchings(l):
        if not l: return [[]]
        # Match the first element with any remaining, recurse
        return [[(l[0], i)] + r for i in l[1:] for r in get_matchings([x for x in l[1:] if x != i])]

    raw_matchings = get_matchings(list(range(10)))
    
    # Format matching as a canonical sorted tuple of tuples for hashing
    def canonicalize(matching):
        return tuple(sorted([tuple(sorted(pair)) for pair in matching]))
    
    # 2. Define the symmetry generators based on the 10-index mapping
    generators = [
        # --- Network A Symmetries ---
        {0:2, 2:0}, # Swap Q1_A <-> K1_A
        {1:3, 3:1}, # Swap Q2_A <-> K2_A
        {0:1, 1:0, 2:3, 3:2}, # Swap Head 1_A <-> Head 2_A
        
        # --- Network B Symmetries ---
        {5:7, 7:5}, # Swap Q1_B <-> K1_B
        {6:8, 8:6}, # Swap Q2_B <-> K2_B
        {5:6, 6:5, 7:8, 8:7}, # Swap Head 1_B <-> Head 2_B
        
        # --- Network A <-> Network B Swap ---
        {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}
    ]
    
    # Ensure all generators map unlisted keys to themselves
    generators = [{i: g.get(i, i) for i in range(10)} for g in generators]

    # Helper to apply a symmetry permutation to a matching
    def apply_perm(matching, p):
        return canonicalize([(p[u], p[v]) for u, v in matching])

    # 3. Find the 26 Orbits (Equivalence Classes)
    orbits = {}
    visited = set()
    
    for m in raw_matchings:
        m_canon = canonicalize(m)
        if m_canon in visited:
            continue
            
        # Discover the full orbit using a BFS through the generators
        orbit_members = set([m_canon])
        queue = [m_canon]
        
        while queue:
            current = queue.pop(0)
            for gen in generators:
                next_m = apply_perm(current, gen)
                if next_m not in orbit_members:
                    orbit_members.add(next_m)
                    queue.append(next_m)
                    
        visited.update(orbit_members)
        
        # Store the first element as the canonical representative
        orbits[m_canon] = len(orbit_members)

    # 4. Format the output into the 26x10 matrix and multiplicities
    P_matrix = []
    multiplicities = []
    
    for canonical_matching, size in orbits.items():
        # Flatten the pairs into a 10-element row
        row = [item for pair in canonical_matching for item in pair]
        P_matrix.append(row)
        multiplicities.append(size)

    return np.array(P_matrix), np.array(multiplicities)

# Run it
P_matrix, multiplicities = generate_bilinear_symmetries()

print(f"Matrix Shape: {P_matrix.shape}") # (26, 10)
print(f"Multiplicities Shape: {multiplicities.shape}") # (26,)
print(f"Total Matchings Accounted For: {multiplicities.sum()}") # 945

# %%

import sparse
import quimb.tensor as qtn

# ... inside your network constructor ...

P, mults = generate_bilinear_symmetries()

# The Sparse Tensor now inherently computes the weighted symmetries!
W_sparse = sparse.COO(
    coords=P.T,              # Shape (10, 26)
    data=mults,              # The 26 weights (ensures mathematical exactness)
    shape=(10,)*10           # The 10-leg bounding box
)

W = qtn.Tensor(W_sparse, inds=(f'r{i}' for i in range(10)), tags=['WICK'])