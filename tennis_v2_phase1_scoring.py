"""Tennis V2 Phase1A — deterministic tennis scoring mechanics.

NO fitted player model and NO match-data reads: input probabilities must come
from an audited pre-match serve/return model in a later Phase1 increment.
Includes simplified singles rules only; unknown tournament formats block.
Uses closed-form deuce probabilities + finite memoized dynamic programming,
not Monte Carlo, a sportsbook value or an Elo-to-match shortcut.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

from tennis_v2_phase1_guards import EvidenceBlocked, FormatEvidence


def _point(p: float, name: str) -> float:
    if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p):
        raise EvidenceBlocked(f"BLOCKED_DATA: {name} must be finite")
    if not 0.000001 <= p <= 0.999999:
        raise EvidenceBlocked(f"BLOCKED_DATA: {name} requires strictly interior probability")
    return float(p)


def _p1_point(p1_serve: float, p2_serve: float, server: int) -> float:
    if server not in (1, 2):
        raise EvidenceBlocked("BLOCKED_FORMAT: server must be 1 or 2")
    return p1_serve if server == 1 else 1.0 - p2_serve


def hold_probability(server_point_win: float) -> float:
    """Exact standard advantage-game win probability for the *server*."""
    p = _point(server_point_win, "server_point_win")
    q = 1.0 - p
    deuce = p * p / (p * p + q * q)
    return p**4 * (1.0 + 4.0*q + 10.0*q*q) + 20.0*p**3*q**3*deuce


def _advantage_two_game_win(p1_game_on_own_serve: float,
                            p1_game_on_p2_serve: float) -> float:
    """Win an extended advantage set from tied >=6-6: each pair splits service."""
    a, b = p1_game_on_own_serve, p1_game_on_p2_serve
    two_wins = a*b
    two_losses = (1-a)*(1-b)
    denom = two_wins + two_losses
    if denom <= 0:
        raise EvidenceBlocked("BLOCKED_FORMAT: degenerate advantage-set process")
    return two_wins/denom


def _tiebreak_server(point_index: int, first_server: int) -> int:
    """Tiebreak serve order: 1, then opponent twice, then alternating pairs."""
    if point_index == 0:
        return first_server
    return (3-first_server) if (((point_index-1)//2) % 2 == 0) else first_server


def tiebreak_win_probability(
    p1_serve_point: float, p2_serve_point: float,
    *, first_server: Literal[1, 2], points_to_win: Literal[7, 10] = 7,
) -> float:
    """P(player 1 wins tiebreak), including exact infinite win-by-two tail."""
    p1 = _point(p1_serve_point, "p1_serve_point")
    p2 = _point(p2_serve_point, "p2_serve_point")
    if first_server not in (1, 2) or points_to_win not in (7, 10):
        raise EvidenceBlocked("BLOCKED_FORMAT: unsupported tiebreak")
    # From >=6-6 or >=9-9, each successive pair is one serve each.
    # The process returns to a tied score or terminates, so this is exact.
    deuce_win = p1*(1.0-p2)
    deuce_lose = (1.0-p1)*p2
    deuce_p1 = deuce_win/(deuce_win+deuce_lose)

    @lru_cache(maxsize=None)
    def solve(w1: int, w2: int) -> float:
        if w1 >= points_to_win and w1-w2 >= 2:
            return 1.0
        if w2 >= points_to_win and w2-w1 >= 2:
            return 0.0
        if w1 == w2 and w1 >= points_to_win-1:
            return deuce_p1
        server = _tiebreak_server(w1+w2, first_server)
        p = _p1_point(p1, p2, server)
        return p*solve(w1+1,w2)+(1.0-p)*solve(w1,w2+1)

    return solve(0, 0)


def match_win_probability(
    p1_serve_point: float,
    p2_serve_point: float,
    *,
    fmt: FormatEvidence,
    first_server: Literal[1, 2] | None = None,
) -> float:
    """P(player 1 wins best-of match); first_server=None averages both equally.

    Format support is explicitly limited to protocol-enumerated singles
    rules. Last-set tie-break logic is year/event-specific upstream;
    do not select it from guessed tournament strings.
    """
    p1 = _point(p1_serve_point, "p1_serve_point")
    p2 = _point(p2_serve_point, "p2_serve_point")
    allowed = {
        (3, "TB7_AT_6_ALL_SETS"),
        (5, "TB7_AT_6_ALL_SETS"),
        (5, "TB10_AT_6_FINAL"),
        (5, "ADVANTAGE_FINAL_SET"),
    }
    if (fmt.best_of,fmt.final_set_rule) not in allowed or first_server not in (None,1,2):
        raise EvidenceBlocked("BLOCKED_FORMAT: unsupported singles scoring")
    need_sets = fmt.best_of//2 + 1
    p1_hold = hold_probability(p1)
    p2_hold = hold_probability(p2)
    p1_break = 1.0-p2_hold

    @lru_cache(maxsize=None)
    def solve(s1: int, s2: int, g1: int, g2: int, server: int) -> float:
        if s1 >= need_sets:
            return 1.0
        if s2 >= need_sets:
            return 0.0
        if g1 >= 6 and g1-g2 >= 2:
            return solve(s1+1,s2,0,0,server)
        if g2 >= 6 and g2-g1 >= 2:
            return solve(s1,s2+1,0,0,server)

        deciding_set = (s1 == need_sets-1 and s2 == need_sets-1)
        if g1 == 6 and g2 == 6:
            if deciding_set and fmt.final_set_rule == "ADVANTAGE_FINAL_SET":
                q = _advantage_two_game_win(p1_hold, p1_break)
                # Every win-by-two termination has an even number of
                # added games, returning next set server to this server.
                return q*solve(s1+1,s2,0,0,server) + (1-q)*solve(s1,s2+1,0,0,server)
            target = 10 if (deciding_set and fmt.final_set_rule == "TB10_AT_6_FINAL") else 7
            q = tiebreak_win_probability(p1,p2,first_server=server,points_to_win=target)
            next_server = 3-server  # receiver of first tiebreak point opens next set
            return q*solve(s1+1,s2,0,0,next_server)+(1-q)*solve(s1,s2+1,0,0,next_server)

        p1_wins_game = p1_hold if server == 1 else p1_break
        next_server = 3-server
        return (p1_wins_game*solve(s1,s2,g1+1,g2,next_server)
                +(1-p1_wins_game)*solve(s1,s2,g1,g2+1,next_server))

    if first_server is not None:
        return solve(0,0,0,0,first_server)
    return 0.5*(solve(0,0,0,0,1)+solve(0,0,0,0,2))
