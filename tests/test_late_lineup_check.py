import unittest
import late_lineup_check as L


class T(unittest.TestCase):
    def test_names(self):
        self.assertEqual(L.norm_name("Alex Ovechkin"), L.norm_name("alex  ovechkin"))
        self.assertEqual(L.norm_name("Jose Ramírez Jr."), L.norm_name("Jose Ramirez"))

    def test_status_map_never_assumes_healthy(self):
        self.assertEqual(L.map_status("Out"), "OUT")
        self.assertEqual(L.map_status("Doubtful"), "DOUBTFUL")
        self.assertEqual(L.map_status("something new"), "QUESTIONABLE")

    def test_espn_injuries(self):
        s = {"injuries": [{"team": {"abbreviation": "TOR"}, "injuries": [
            {"status": "Out", "athlete": {"displayName": "John Tavares"}},
            {"status": "Day-To-Day", "athlete": {"displayName": "Auston Matthews"}}]}]}
        self.assertEqual(L.parse_espn_injuries(s), {"TOR": {"johntavares": "OUT", "austonmatthews": "QUESTIONABLE"}})

    def test_nhl_official(self):
        land = {"awayTeam": {"abbrev": "UTA", "id": 59}, "homeTeam": {"abbrev": "TOR", "id": 10},
                "rosterSpots": [{"teamId": 10, "firstName": {"default": "Auston"}, "lastName": {"default": "Matthews"}}]}
        rail = {"gameInfo": {"homeTeam": {"scratches": [{"firstName": {"default": "John"}, "lastName": {"default": "Tavares"}}]}, "awayTeam": {"scratches": []}}}
        posted, t = L.parse_nhl_official(land, rail)
        self.assertTrue(posted)
        self.assertEqual(t["TOR"]["austonmatthews"], "IN")
        self.assertEqual(t["TOR"]["johntavares"], "OUT")

    def test_nothing_posted_changes_nothing(self):
        land = {"awayTeam": {"abbrev": "UTA", "id": 59}, "homeTeam": {"abbrev": "TOR", "id": 10}, "rosterSpots": []}
        posted, t = L.parse_nhl_official(land, {"gameInfo": {"homeTeam": {"scratches": []}, "awayTeam": {"scratches": []}}})
        self.assertFalse(posted)
        rows = [{"player": "Auston Matthews", "team": "TOR", "mean": 3.0}]
        out = L.apply_overlay("nhl", rows, {"lineup_posted": posted, "teams": t})
        self.assertEqual(out[0]["mean"], 3.0)
        self.assertEqual(out[0]["mean_early"], 3.0)
        self.assertEqual(out[0]["lineup_label"], "NO_NEW_INFO")

    def test_out_is_zero_and_early_forecast_kept(self):
        gs = {"lineup_posted": False, "teams": {"TOR": {"johntavares": "OUT"}}}
        out = L.apply_overlay("nhl", [{"player": "John Tavares", "team": "TOR", "mean": 2.0}], gs)[0]
        self.assertEqual(out["mean"], 0.0)
        self.assertEqual(out["mean_early"], 2.0)
        self.assertEqual(out["lineup_label"], "CONFIRMED_OUT")

    def test_posted_lineup_confirms_and_drops_missing(self):
        gs = {"lineup_posted": True, "teams": {"TOR": {"austonmatthews": "IN"}}}
        rows = [{"player": "Auston Matthews", "team": "TOR", "mean": 2.0}, {"player": "Spare Guy", "team": "TOR", "mean": 1.0}]
        a, b = L.apply_overlay("nhl", rows, gs)
        self.assertAlmostEqual(a["mean"], 2.0 * 1.058)
        self.assertEqual(b["mean"], 0.0)
        self.assertEqual(b["lineup_label"], "NOT_IN_POSTED_LINEUP")

    def test_disabled_sport_is_untouched(self):
        gs = {"lineup_posted": True, "teams": {"X": {"a": "OUT"}}}
        out = L.apply_overlay("nba", [{"player": "A", "team": "X", "mean": 5.0}], gs)[0]
        self.assertEqual(out["mean"], 5.0)

    def test_merge_never_downgrades_out(self):
        m = L.merge({"T": {"a": "OUT"}}, {"T": {"a": "IN"}})
        self.assertEqual(m["T"]["a"], "OUT")


if __name__ == "__main__":
    unittest.main()
