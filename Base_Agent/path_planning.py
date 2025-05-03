import numpy as np
from queue import PriorityQueue, Queue
from typing import List, Tuple, Dict, Set
import heapq
import random

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Centralized Algorithms

def apply_minimax_algorithm(sources: List[Tuple[int, int]], 
                          targets: List[Tuple[int, int]],
                          depth: int = 3) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Apply Minimax algorithm for centralized path planning.
    Uses game theory to minimize maximum distance any agent needs to travel.
    """
    if not sources or not targets:
        print("Error: No sources or targets provided")
        return []

    # Ensure we have valid inputs
    if len(sources) > len(targets):
        sources = sources[:len(targets)]
    elif len(targets) > len(sources):
        targets = targets[:len(sources)]
    
    def evaluate_assignment(assignment):
        if not assignment:  # Safety check
            return float('-inf')
        max_distance = float('-inf')
        for source_idx, target in assignment:
            distance = manhattan_distance(sources[source_idx], target)
            max_distance = max(max_distance, distance)
        return -max_distance  # Negative because minimax tries to maximize
    
    def minimax(depth, assignments, maximizing):
        if depth == 0 or len(assignments) == len(sources):
            return evaluate_assignment(assignments)
            
        if maximizing:
            max_eval = float('-inf')
            for i, source in enumerate(sources):
                if not any(src_idx == i for src_idx, _ in assignments):
                    for target in targets:
                        if not any(tgt == target for _, tgt in assignments):
                            assignments.append((i, target))
                            eval = minimax(depth - 1, assignments, False)
                            assignments.pop()
                            max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for i, source in enumerate(sources):
                if not any(src_idx == i for src_idx, _ in assignments):
                    for target in targets:
                        if not any(tgt == target for _, tgt in assignments):
                            assignments.append((i, target))
                            eval = minimax(depth - 1, assignments, True)
                            assignments.pop()
                            min_eval = min(min_eval, eval)
            return min_eval
    
    # Find best initial assignment
    best_assignments = []
    best_value = float('-inf')
    
    for i, source in enumerate(sources):
        for target in targets:
            assignments = [(i, target)]
            value = minimax(depth - 1, assignments, False)
            if value > best_value:
                best_value = value
                best_assignments = assignments.copy()
    
    # Validate assignments
    if not best_assignments:
        print("Warning: No valid assignments found")
        # Create simple 1-to-1 assignments as fallback
        for i in range(min(len(sources), len(targets))):
            best_assignments.append((i, targets[i]))
    
    return best_assignments

def apply_alpha_beta_algorithm(sources: List[Tuple[int, int]], 
                             targets: List[Tuple[int, int]],
                             depth: int = 4) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Apply Alpha-Beta pruning for more efficient centralized path planning.
    Similar to minimax but with pruning of unnecessary branches.
    """
    def evaluate_assignment(assignment):
        total_distance = 0
        for source_idx, target in assignment:
            distance = manhattan_distance(sources[source_idx], target)
            total_distance += distance
        return -total_distance  # Negative because alpha-beta tries to maximize
    
    def alpha_beta(depth, assignments, alpha, beta, maximizing):
        if depth == 0 or len(assignments) == len(sources):
            return evaluate_assignment(assignments)
            
        if maximizing:
            value = float('-inf')
            for i, source in enumerate(sources):
                if not any(src_idx == i for src_idx, _ in assignments):
                    for target in targets:
                        if not any(tgt == target for _, tgt in assignments):
                            assignments.append((i, target))
                            value = max(value, alpha_beta(depth - 1, assignments, alpha, beta, False))
                            assignments.pop()
                            alpha = max(alpha, value)
                            if alpha >= beta:
                                break
            return value
        else:
            value = float('inf')
            for i, source in enumerate(sources):
                if not any(src_idx == i for src_idx, _ in assignments):
                    for target in targets:
                        if not any(tgt == target for _, tgt in assignments):
                            assignments.append((i, target))
                            value = min(value, alpha_beta(depth - 1, assignments, alpha, beta, True))
                            assignments.pop()
                            beta = min(beta, value)
                            if beta <= alpha:
                                break
            return value
    
    # Find best initial assignment
    best_assignments = []
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for i, source in enumerate(sources):
        for target in targets:
            assignments = [(i, target)]
            value = alpha_beta(depth - 1, assignments, alpha, beta, False)
            if value > best_value:
                best_value = value
                best_assignments = assignments.copy()
            alpha = max(alpha, best_value)
    
    return best_assignments

def apply_expectimax_algorithm(sources: List[Tuple[int, int]], 
                             targets: List[Tuple[int, int]],
                             depth: int = 3) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Apply Expectimax algorithm for centralized path planning under uncertainty.
    Considers probabilistic outcomes for more robust planning.
    """
    if not sources or not targets:
        print("Error: No sources or targets provided")
        return []

    # Ensure we have valid inputs
    if len(sources) > len(targets):
        sources = sources[:len(targets)]
    elif len(targets) > len(sources):
        targets = targets[:len(sources)]
    
    def evaluate_assignment(assignment):
        if not assignment:  # Safety check
            return float('-inf')
        total_distance = 0
        for source_idx, target in assignment:
            distance = manhattan_distance(sources[source_idx], target)
            total_distance += distance
        return -total_distance
    
    def expectimax(depth, assignments, is_max):
        if depth == 0 or len(assignments) == len(sources):
            return evaluate_assignment(assignments)
            
        if is_max:
            value = float('-inf')
            for i, source in enumerate(sources):
                if not any(src_idx == i for src_idx, _ in assignments):
                    for target in targets:
                        if not any(tgt == target for _, tgt in assignments):
                            assignments.append((i, target))
                            value = max(value, expectimax(depth - 1, assignments, False))
                            assignments.pop()
            return value
        else:
            # Chance node: consider all possible outcomes with equal probability
            values = []
            count = 0
            for i, source in enumerate(sources):
                if not any(src_idx == i for src_idx, _ in assignments):
                    for target in targets:
                        if not any(tgt == target for _, tgt in assignments):
                            assignments.append((i, target))
                            values.append(expectimax(depth - 1, assignments, True))
                            assignments.pop()
                            count += 1
            return sum(values) / count if count > 0 else float('-inf')
    
    # Find best initial assignment
    best_assignments = []
    best_value = float('-inf')
    
    for i, source in enumerate(sources):
        for target in targets:
            assignments = [(i, target)]
            value = expectimax(depth - 1, assignments, False)
            if value > best_value:
                best_value = value
                best_assignments = assignments.copy()
    
    # Validate assignments
    if not best_assignments:
        print("Warning: No valid assignments found")
        # Create simple 1-to-1 assignments as fallback
        for i in range(min(len(sources), len(targets))):
            best_assignments.append((i, targets[i]))
    
    return best_assignments

# Distributed Algorithms

def apply_gradient_based_algorithm(sources: List[Tuple[int, int]], 
                                 targets: List[Tuple[int, int]]) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Apply gradient-based algorithm for distributed path planning.
    Each agent follows the gradient towards closest target while avoiding conflicts.
    """
    if not sources or not targets:
        print("Error: No sources or targets provided")
        return []

    assignments = []
    unassigned_targets = set(targets)
    
    # Ensure we have valid inputs
    if len(sources) > len(targets):
        sources = sources[:len(targets)]
    elif len(targets) > len(sources):
        targets = targets[:len(sources)]
    
    def calculate_potential(pos, target):
        """Calculate potential field value at a position relative to a target"""
        distance = manhattan_distance(pos, target)
        if distance == 0:
            return float('inf')  # Avoid division by zero
        return 1.0 / distance

    # For each source, find the best target based on potential field
    for i, source in enumerate(sources):
        if not unassigned_targets:  # Safety check
            break
            
        best_target = None
        best_potential = float('-inf')
        
        for target in unassigned_targets:
            potential = calculate_potential(source, target)
            if potential > best_potential:
                best_potential = potential
                best_target = target
        
        if best_target:  # Safety check
            assignments.append((i, best_target))
            unassigned_targets.remove(best_target)
    
    # Validate assignments
    if not assignments:
        print("Warning: No valid assignments found")
        # Create simple 1-to-1 assignments as fallback
        for i in range(min(len(sources), len(targets))):
            assignments.append((i, targets[i]))
    
    return assignments

def apply_cellular_automata_algorithm(sources: List[Tuple[int, int]], 
                                    targets: List[Tuple[int, int]]) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Apply cellular automata algorithm for distributed path planning.
    Agents make decisions based on local neighborhood rules.
    """
    if not sources or not targets:
        print("Error: No sources or targets provided")
        return []

    assignments = []
    unassigned_targets = set(targets)
    
    # Ensure we have valid inputs
    if len(sources) > len(targets):
        sources = sources[:len(targets)]
    elif len(targets) > len(sources):
        targets = targets[:len(sources)]
    
    def get_neighborhood_score(source, target, existing_assignments):
        """Calculate score based on neighborhood rules"""
        base_score = -manhattan_distance(source, target)
        
        # Add penalty for proximity to other assignments
        for _, assigned_target in existing_assignments:
            if manhattan_distance(target, assigned_target) < 2:
                base_score -= 5
                
        return base_score
    
    # Assign agents to targets based on neighborhood rules
    for i, source in enumerate(sources):
        if not unassigned_targets:  # Safety check
            break
            
        best_target = None
        best_score = float('-inf')
        
        for target in unassigned_targets:
            score = get_neighborhood_score(source, target, assignments)
            if score > best_score:
                best_score = score
                best_target = target
        
        if best_target:  # Safety check
            assignments.append((i, best_target))
            unassigned_targets.remove(best_target)
    
    # Validate assignments
    if not assignments:
        print("Warning: No valid assignments found")
        # Create simple 1-to-1 assignments as fallback
        for i in range(min(len(sources), len(targets))):
            assignments.append((i, targets[i]))
    
    return assignments

def apply_astar_greedy_local(sources: List[Tuple[int, int]], 
                            targets: List[Tuple[int, int]]) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Apply A*/Greedy algorithm for local distributed path planning.
    Each agent independently plans its path using A* or greedy approach.
    """
    assignments = []
    unassigned_targets = set(targets)
    
    for i, source in enumerate(sources):
        if not unassigned_targets:
            break
            
        # Use A* heuristic for local decision
        best_target = None
        min_cost = float('inf')
        
        for target in unassigned_targets:
            # g(n) = current distance
            g_cost = manhattan_distance(source, target)
            
            # h(n) = estimate of congestion and future cost
            h_cost = sum(manhattan_distance(target, t) for t in unassigned_targets if t != target)
            h_cost = h_cost / len(unassigned_targets) if unassigned_targets else 0
            
            total_cost = g_cost + 0.5 * h_cost  # Weight heuristic less for more local decisions
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_target = target
        
        if best_target:
            assignments.append((i, best_target))
            unassigned_targets.remove(best_target)
    
    return assignments

def apply_heuristic_movement(sources: List[Tuple[int, int]], 
                           targets: List[Tuple[int, int]]) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Apply heuristic-based movement for distributed path planning.
    Uses multiple heuristics for robust local decision making.
    """
    assignments = []
    unassigned_targets = set(targets)
    
    def calculate_heuristic_score(source, target, existing_assignments):
        # Distance heuristic
        distance_score = manhattan_distance(source, target)
        
        # Congestion heuristic
        congestion = sum(1 for _, t in existing_assignments if manhattan_distance(target, t) < 3)
        congestion_score = congestion * 2
        
        # Target clustering heuristic
        clustering = sum(1 for t in unassigned_targets if manhattan_distance(target, t) < 3)
        clustering_score = -clustering  # Negative because clustering is good
        
        # Combine heuristics
        return distance_score + congestion_score + clustering_score
    
    for i, source in enumerate(sources):
        if not unassigned_targets:
            break
            
        best_target = None
        best_score = float('inf')
        
        for target in unassigned_targets:
            score = calculate_heuristic_score(source, target, assignments)
            if score < best_score:
                best_score = score
                best_target = target
        
        if best_target:
            assignments.append((i, best_target))
            unassigned_targets.remove(best_target)
    
    return assignments