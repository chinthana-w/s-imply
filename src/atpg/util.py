
# Minimal utility implementations for PODEM

from collections import deque
from src.util.struct import LogicValue, GateType

def calculate_scoap(circuit, total_gates):
    # No-op for now
    pass

def calculate_distance_to_primary_inputs(circuit, total_gates):
    # BFS from PIs
    dists = {}
    queue = deque()
    
    for i in range(1, total_gates + 1):
        if circuit[i].type == GateType.INPT:
            dists[i] = 0
            queue.append(i)
            
    # Forward traversal? NO.
    # We want distance FROM PI.
    # BFS forward from PIs.
    
    while queue:
        u = queue.popleft()
        d = dists[u]
        
        if circuit[u].fot:
            for v in circuit[u].fot:
                if v not in dists: # First visit = shortest path
                    dists[v] = d + 1
                    queue.append(v)
    return dists

def get_topological_order(circuit, total_gates):
    """Kahn's or level-based topological sort."""
    in_degree = [0] * (total_gates + 1)
    for i in range(1, total_gates + 1):
        if circuit[i].type != 0:
            in_degree[i] = len(circuit[i].fin)
            
    order = []
    # Start with nodes that have 0 in-degree (PIs)
    queue = deque([i for i in range(1, total_gates + 1) if circuit[i].type != 0 and in_degree[i] == 0])
    
    while queue:
        u = queue.popleft()
        order.append(u)
        
        for v in circuit[u].fot:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    return order


def calculate_distance_to_primary_outputs(circuit, total_gates):
    # Backwards BFS from POs
    dists = {}
    queue = deque()
    
    # Find POs
    for i in range(1, total_gates + 1):
        if circuit[i].type != 0 and circuit[i].nfo == 0:
            dists[i] = 0
            queue.append(i)
            
    # BFS backwards logic requires reverse graph.
    # circuit only has fin (inputs). Ideal for backward traversal.
    
    visited = set(dists.keys())
    
    while queue:
        u = queue.popleft()
        d = dists[u]
        
        for fin in circuit[u].fin:
            if fin not in visited:
                dists[fin] = d + 1
                visited.add(fin)
                queue.append(fin)
                
    return dists

def get_x_fanin(circuit, gate_id, distances):
    """
    Pick a fanin with value X.
    Heuristic: Pick the one with EASIEST controllability?
    Or standard PODEM might use specific SCOAP.
    Here we use distance to PI (shorter = easier to control?).
    """
    candidates = []
    gate = circuit[gate_id]
    
    for fin in gate.fin:
        if circuit[fin].val == LogicValue.XD:
            candidates.append(fin)
            
    if not candidates:
        return -1
        
    # Sort by distance (asc)
    # If distances not available, default 0
    candidates.sort(key=lambda x: distances.get(x, float('inf')))
    
    return candidates[0]
