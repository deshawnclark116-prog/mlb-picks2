"""Research-only Phase1A safety guarantees. Uses no real or 2025 match outcomes."""
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from tennis_v2_phase1_guards import (
    EvidenceBlocked, FinalizedResult, FormatEvidence, HistoricalResult,
    IdentityEvidence, SurfaceEvidence, forward_asof, historical_asof,
    pre_match_gate,
)

UTC = timezone.utc


def dt(day=5, hour=10):
    return datetime(2026, 10, day, hour, tzinfo=UTC)


def h(key, day, *, event="2024_A", tour="atp", decided=True, completed=True):
    return HistoricalResult(
        tour=tour, event_key=event, match_key=key,
        tournament_start=f"2024-01-{day:02d}", decided=decided,
        completed=completed, winner_id=f"{key}_w", loser_id=f"{key}_l"
    )


def controls():
    a = IdentityEvidence("atp", "provider_a", "tml_a", "EXACT", True, 1, dt(4))
    b = IdentityEvidence("atp", "provider_b", "tml_b", "NORMALIZED_EXACT", True, 1, dt(4))
    s = SurfaceEvidence("atp", "2026_T", 2026, "Clay", "KNOWN_EXACT", dt(4), 2026)
    f = FormatEvidence(3, "TB7_AT_6_ALL_SETS")
    return dict(tour="atp", event_key="2026_T", season=2026,
                decision_at_utc=dt(), player1=a, player2=b, surface=s, fmt=f)


class HistoricalChronologyTests(unittest.TestCase):
    def test_embargo_exact_boundary(self):
        rows = [h("prior", 1), h("in_embargo", 2)]
        out = historical_asof(rows, target_tour="atp", target_event_key="2024_T",
                             target_tournament_start="2024-01-29", purpose="rating")
        self.assertEqual([r.match_key for r in out], ["prior"])

    def test_same_event_later_round_never_appears(self):
        rows = [h("F", 1, event="2024_T"), h("R128", 1, event="2024_T"), h("old", 1)]
        out = historical_asof(rows, target_tour="atp", target_event_key="2024_T",
                             target_tournament_start="2024-03-01", purpose="rating")
        self.assertEqual([r.match_key for r in out], ["old"])

    def test_row_order_does_not_affect_forecast_history(self):
        rows = [h("z", 1), h("a", 1, event="2024_B"), h("x", 2, event="2024_C")]
        args = dict(target_tour="atp", target_event_key="2024_T",
                    target_tournament_start="2024-03-01", purpose="rating")
        left = historical_asof(rows, **args)
        right = historical_asof(list(reversed(rows)), **args)
        self.assertEqual(left, right)

    def test_appending_future_outcome_does_not_change_history(self):
        rows = [h("old", 1)]
        args = dict(target_tour="atp", target_event_key="2024_T",
                    target_tournament_start="2024-02-01", purpose="rating")
        first = historical_asof(rows, **args)
        future = replace(h("future", 1, event="2025_T"), tournament_start="2025-01-01")
        self.assertEqual(first, historical_asof(rows + [future], **args))

    def test_retirment_rating_not_performance(self):
        rows = [h("ret", 1, completed=False)]
        args = dict(target_tour="atp", target_event_key="2024_T",
                    target_tournament_start="2024-03-01")
        self.assertEqual(len(historical_asof(rows, purpose="rating", **args)), 1)
        self.assertEqual(len(historical_asof(rows, purpose="performance", **args)), 0)

    def test_walkover_never_consumed(self):
        rows = [h("wo", 1, decided=False, completed=False)]
        self.assertEqual(historical_asof(
            rows, target_tour="atp", target_event_key="2024_T",
            target_tournament_start="2024-03-01", purpose="rating"), ())

    def test_cross_tour_results_do_not_mix(self):
        rows = [h("atp", 1), h("wta", 1, event="2024_W", tour="wta")]
        out = historical_asof(rows, target_tour="atp", target_event_key="2024_T",
                             target_tournament_start="2024-03-01", purpose="rating")
        self.assertEqual([r.match_key for r in out], ["atp"])

    def test_duplicate_ids_fail_closed(self):
        with self.assertRaisesRegex(EvidenceBlocked, "duplicate match identity"):
            historical_asof([h("dup", 1), h("dup", 2)],
                            target_tour="atp", target_event_key="2024_T",
                            target_tournament_start="2024-03-01", purpose="rating")

    def test_changing_embargo_needs_new_protocol(self):
        with self.assertRaisesRegex(EvidenceBlocked, "protocol-locked"):
            historical_asof([], target_tour="atp", target_event_key="2024_T",
                            target_tournament_start="2024-03-01",
                            purpose="rating", embargo_days=7)

    def test_noncanonical_dates_fail_closed(self):
        with self.assertRaisesRegex(EvidenceBlocked, "date"):
            historical_asof([h("x", 1)], target_tour="atp", target_event_key="2024_T",
                            target_tournament_start="2024/03/01", purpose="rating")


class ForwardChronologyTests(unittest.TestCase):
    def res(self, name, finished, retrieved, *, event="old", tour="atp"):
        return FinalizedResult(tour, event, name, finished, retrieved)

    def test_only_observed_prior_results_are_eligible(self):
        rows = [
            self.res("old", dt(4), dt(4, 11)),
            self.res("future", dt(5, 10), dt(5, 12)),
            self.res("same", dt(4), dt(4, 11), event="T"),
            self.res("other_tour", dt(4), dt(4, 11), tour="wta"),
        ]
        found = forward_asof(rows, target_tour="atp", target_event_key="T",
                            scheduled_start_utc=dt(5, 14),
                            forecast_created_at_utc=dt(5, 11))
        self.assertEqual([x.match_key for x in found], ["old"])

    def test_forecast_after_start_is_blocked(self):
        with self.assertRaisesRegex(EvidenceBlocked, "decision is not before"):
            forward_asof([], target_tour="atp", target_event_key="T",
                         scheduled_start_utc=dt(5, 10),
                         forecast_created_at_utc=dt(5, 11))

    def test_naive_timestamps_blocked(self):
        with self.assertRaisesRegex(EvidenceBlocked, "UTC-aware"):
            forward_asof([], target_tour="atp", target_event_key="T",
                         scheduled_start_utc=datetime(2026, 10, 5, 14),
                         forecast_created_at_utc=dt())

    def test_result_retrieved_before_finalization_blocked(self):
        with self.assertRaisesRegex(EvidenceBlocked, "before finalization"):
            forward_asof([self.res("bad", dt(5, 12), dt(5, 11))],
                         target_tour="atp", target_event_key="T",
                         scheduled_start_utc=dt(5, 14), forecast_created_at_utc=dt(5, 13))

    def test_permutation_stable(self):
        rows = [self.res("b", dt(4), dt(4, 11)),
                self.res("a", dt(4), dt(4, 11))]
        k = dict(target_tour="atp", target_event_key="T",
                 scheduled_start_utc=dt(5, 14), forecast_created_at_utc=dt(5, 11))
        self.assertEqual(forward_asof(rows, **k), forward_asof(rows[::-1], **k))


class IdentitySurfaceFormatTests(unittest.TestCase):
    def test_valid_evidence_is_accepted(self):
        self.assertTrue(pre_match_gate(**controls()).allowed)

    def test_serban_surname_guess_blocked(self):
        c = controls()
        c["player1"] = replace(c["player1"], provider_id="Isabella Maria Serban",
                               tml_player_id="Raluka Serban",
                               match_class="SURNAME_FALLBACK_GUESS")
        out = pre_match_gate(**c)
        self.assertFalse(out.allowed)
        self.assertIn("BLOCKED_IDENTITY", out.blocked_reasons)

    def test_unverified_or_ambiguous_crosswalk_blocked(self):
        for changed in (
            dict(independent_crosswalk_verified=False),
            dict(candidates=2),
            dict(match_class="AMBIGUOUS"),
        ):
            with self.subTest(changed=changed):
                c = controls()
                c["player2"] = replace(c["player2"], **changed)
                self.assertIn("BLOCKED_IDENTITY", pre_match_gate(**c).blocked_reasons)

    def test_hard_fallback_does_not_pass_as_known(self):
        c = controls()
        c["surface"] = replace(c["surface"], surface="Hard", match_class="HARD_FALLBACK")
        self.assertIn("BLOCKED_SURFACE", pre_match_gate(**c).blocked_reasons)

    def test_no_future_season_surface_inference(self):
        c = controls()
        c["surface"] = replace(c["surface"], match_class="HISTORICAL_TOURNEY_INFERENCE",
                               evidence_season=2027)
        self.assertIn("BLOCKED_SURFACE", pre_match_gate(**c).blocked_reasons)

    def test_valid_historical_surface_must_precede_season(self):
        c = controls()
        c["surface"] = replace(c["surface"], match_class="HISTORICAL_TOURNEY_INFERENCE",
                               evidence_season=2026)
        self.assertIn("BLOCKED_SURFACE", pre_match_gate(**c).blocked_reasons)

    def test_mapping_made_after_decision_is_blocked(self):
        c = controls()
        c["player1"] = replace(c["player1"], verified_at_utc=dt(5, 11))
        self.assertIn("BLOCKED_TIMING", pre_match_gate(**c).blocked_reasons)

    def test_unsupported_format_is_blocked(self):
        c = controls()
        c["fmt"] = FormatEvidence(3, "MATCH_TB10_THIRD_SET")
        self.assertIn("BLOCKED_FORMAT", pre_match_gate(**c).blocked_reasons)

    def test_surfaces_must_use_correct_event(self):
        c = controls()
        c["surface"] = replace(c["surface"], event_key="another_event")
        self.assertIn("BLOCKED_SURFACE", pre_match_gate(**c).blocked_reasons)


if __name__ == "__main__":
    unittest.main()
