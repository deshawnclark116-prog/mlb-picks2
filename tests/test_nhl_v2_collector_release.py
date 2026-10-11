"""NHL V2 main scheduled research collector: enforce separate-checkout commit and zero false picks."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/nhl_v2_prospective_delivery.yml"

class TestNhlV2CollectorRelease(unittest.TestCase):
    def test_research_checkout_configured_before_immutable_commit(self):
        s = WF.read_text()
        commit_step = s.index('git commit -m "NHL V2 genuine new cutoff-window count research receipts"')
        research_start = s.index('name: Generate ONLY truly on-time count forecasts')
        config = s.index('git config user.name "github-actions[bot]"', research_start)
        email = s.index('git config user.email "41898282+github-actions[bot]@users.noreply.github.com"', research_start)
        assert research_start < config < email < commit_step
        assert 'git add -A nhl_models/nhl_player_outcome_v2/phase1a_sog_forward' in s
        assert 'git push origin HEAD:feature/nhl-v2-nfl-style-projection-board' in s

    def test_no_betting_claim_and_immutable_forecast_source(self):
        s=WF.read_text()
        assert 'python nhl_v2_phase1a_sog_forward.py verify' in s
        assert 'python -u nhl_v2_phase1a_sog_forward.py run' in s
        assert 'assert not source_errors' in s
        assert 'original/nhl_v2_projection_board_publish.py' in s
        assert 'assert v[\'actual_player_book_lines_verified\'] is False' in s
        assert 'assert v[\'dressed_lineups_verified\'] is False' in s
        assert 'docs/nhl_predictions.json' in s
        assert 'docs/nhl_picks_log.jsonl' in s
        assert "if: github.ref == 'refs/heads/main'" in s
        assert 'cancel-in-progress: false' in s

if __name__=="__main__":
    unittest.main()
