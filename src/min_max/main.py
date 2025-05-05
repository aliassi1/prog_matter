# ---------------------------------------------------------------
# minimax_grid.py  (Python ≥3.9, no external deps)
# Demonstrates depth‑limited Minimax + α‑β pruning (with optional
# Expectimax) in a simple two‑team grid world.
# ---------------------------------------------------------------
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Sequence

# -------------------- TYPE ALIASES -----------------------------------------
Position = Tuple[int, int]   # (row, col) on the grid
Move     = Tuple[int, int]   # Δ(row), Δ(col)

# -------------------- CONFIGURATION ----------------------------------------
GRID_N, GRID_M   = 15, 15          # grid size
DEPTH            = 2               # #plies the search sees ahead
USE_EXPECTIMAX   = False           # treat Red as stochastic if True
CARDINALS        = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N,S,W,E
STAY             = (0, 0)          # wait action

# -------------------- STATE DEFINITION -------------------------------------
@dataclass(frozen=True, slots=True)
class State:
    """
    Immutable game position.
    `turn` is either "blue" (Max player) or "red" (Min/Chance player).
    """
    blue: Tuple[Position, ...]
    red:  Tuple[Position, ...]
    turn: str                       # whose turn

    # -------------- helpers -----------------
    def in_bounds(self, p: Position) -> bool:
        return 0 <= p[0] < GRID_N and 0 <= p[1] < GRID_M

    def occupied(self) -> set[Position]:
        return set(self.blue) | set(self.red)

    # -------------- successor generation ----
    def successors(self) -> List[Tuple[Move, "State"]]:
        """
        All single‑agent moves for the side to move.
        Each successor is (move_vector, new_state).
        """
        succ: List[Tuple[Move, State]] = []
        if self.turn == "blue":
            agents, opp, nxt_turn = self.blue, self.red, "red"
        else:
            agents, opp, nxt_turn = self.red, self.blue, "blue"

        occ = self.occupied()
        for idx, pos in enumerate(agents):
            for dx, dy in CARDINALS + [STAY]:
                nxt = (pos[0] + dx, pos[1] + dy)
                if not self.in_bounds(nxt):
                    continue
                if nxt in occ and nxt != pos:           # cannot move onto another agent
                    continue

                # build new agent tuple
                new_agents = list(agents)
                new_agents[idx] = nxt
                new_agents_t = tuple(new_agents)

                new_state = (
                    State(new_agents_t, opp, nxt_turn)
                    if self.turn == "blue"
                    else State(opp, new_agents_t, nxt_turn)
                )
                succ.append(((dx, dy), new_state))

        return succ

# -------------------- HEURISTIC EVALUATION ----------------------------------
def evaluate(state: State,
             blue_targets: Sequence[Position],
             red_targets:  Sequence[Position]) -> float:
    """
    Positive scores are good for Blue (Max).
    Simple distance heuristic with a mild collision penalty.
    """
    blue_score = 0.0
    for pos, tgt in zip(state.blue, blue_targets):
        blue_score -= abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])  # smaller dist = better

    red_score = 0.0
    for pos, tgt in zip(state.red, red_targets):
        red_score += abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])   # Blue likes Red to be far

    # discourage overlaps
    if len(set(state.blue)) != len(state.blue):
        blue_score -= 1_000
    if len(set(state.red)) != len(state.red):
        red_score += 1_000

    return blue_score + red_score

# -------------------- MINIMAX / EXPECTIMAX ----------------------------------
def minimax(state: State,
            depth: int,
            alpha: float,
            beta: float,
            blue_tgts: Sequence[Position],
            red_tgts:  Sequence[Position]) -> Tuple[float, Move]:
    """
    Returns (value, best_single_agent_move) for the side to play.
    NOTE: depth counts *plies* (a single agent action), not full turns.
    """
    # terminal node or horizon reached
    if depth == 0:
        return evaluate(state, blue_tgts, red_tgts), STAY

    # ---------------- MAX (Blue) ----------------
    if state.turn == "blue":
        best_val, best_mv = -math.inf, STAY
        for mv, nxt in state.successors():
            val, _ = minimax(nxt, depth - 1, alpha, beta, blue_tgts, red_tgts)
            if val > best_val:
                best_val, best_mv = val, mv
            alpha = max(alpha, best_val)
            if beta <= alpha:          # α‑β cutoff
                break
        return best_val, best_mv

    # ---------------- MIN or EXPECTI‑MAX (Red) --
    if USE_EXPECTIMAX:
        total = 0.0
        legal = state.successors()
        for _, nxt in legal:
            val, _ = minimax(nxt, depth - 1, alpha, beta, blue_tgts, red_tgts)
            total += val
        return total / len(legal), STAY

    # adversarial MIN
    best_val, best_mv = math.inf, STAY
    for mv, nxt in state.successors():
        val, _ = minimax(nxt, depth - 1, alpha, beta, blue_tgts, red_tgts)
        if val < best_val:
            best_val, best_mv = val, mv
        beta = min(beta, best_val)
        if beta <= alpha:              # α‑β cutoff
            break
    return best_val, best_mv

# -------------------- MAIN LOOP --------------------------------------------
def play_game(seed: int | None = None) -> None:
    """
    Initialise a tiny scenario and let the AI pick one agent move per ply.
    Useful for sanity‑testing the search.
    """
    random.seed(seed)
    blue_start = [(1, 1), (1, 2), (2, 1), (2, 2)]
    red_start  = [(5, 5), (5, 6), (6, 5), (6, 6)]
    blue_tgts  = [(10, 10), (10, 11), (11, 10), (11, 11)]
    red_tgts   = [(7, 7), (7, 8), (8, 7), (8, 8)]

    state = State(tuple(blue_start), tuple(red_start), "blue")

    for ply in range(100):
        print(f"\nPly {ply + 1} | {state.turn.upper()} to move")
        print("Blue:", state.blue)
        print("Red :", state.red)

        val, best_move = minimax(state, DEPTH,
                                 -math.inf, math.inf,
                                 blue_tgts, red_tgts)
        print("Chosen move:", best_move, "| heuristic =", val)

        # apply the chosen move to the *first* agent of the side to play
        idx = 0
        if state.turn == "blue":
            new_pos  = (state.blue[idx][0] + best_move[0],
                        state.blue[idx][1] + best_move[1])
            blue_new = list(state.blue); blue_new[idx] = new_pos
            state = State(tuple(blue_new), state.red, "red")
        else:
            new_pos  = (state.red[idx][0] + best_move[0],
                        state.red[idx][1] + best_move[1])
            red_new  = list(state.red); red_new[idx] = new_pos
            state = State(state.blue, tuple(red_new), "blue")

        # win conditions (very naïve)
        if all(p == t for p, t in zip(state.blue, blue_tgts)):
            print("\nBLUE reached all targets — Blue wins!")
            break
        if all(p == t for p, t in zip(state.red, red_tgts)):
            print("\nRED reached block points — Red wins!")
            break

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    play_game()
