"""The NHL V2 deployment must preserve the original engine and durable source receipts."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
WF=ROOT/".github/workflows/nhl_v2_prospective_delivery.yml"
WORKER=ROOT/"nhl_v2_cutoff_window_worker.py"
ENGINE="nhl_v2_phase1a_sog_forward.py"


class TestNhlV2CollectorRelease(unittest.TestCase):
    def test_immutable_research_checkout_is_committed_with_identity(self):
        wf=WF.read_text()
        op=WORKER.read_text()
        assert "ref: feature/nhl-v2-nfl-style-projection-board" in wf
        assert "nhl_v2_cutoff_window_worker.py" in wf
        start=op.index("def git_push(")
        name=op.index('["git", "config", "user.name", "github-actions[bot]"]',start)
        email=op.index('"41898282+github-actions[bot]@users.noreply.github.com"',start)
        add=op.index('"nhl_models/nhl_player_outcome_v2/phase1a_sog_forward"',start)
        commit=op.index('["git", "commit", "-m"',start)
        assert start<name<email<add<commit
        assert '"HEAD:" + BRANCH' in op
        assert 'command(["git", "rebase", "origin/" + BRANCH]' in op

    def test_model_provenance_and_no_bettable_fake_lines(self):
        wf=WF.read_text()
        op=WORKER.read_text()
        assert "Continuously cover NEXT 108 minutes" in wf
        assert "duration-min 108" in wf
        assert "verify_original(original)" in op
        assert '"nhl_v2_phase1a_sog_forward.py", "run"' in op
        assert '"nhl_v2_phase1a_sog_forward.py", "verify"' in op
        assert "nhl_v2_projection_board_publish.py" in op
        assert '("actual_player_book_lines_verified", False)' in op
        assert '("dressed_lineups_verified", False)' in op
        assert '("not_a_betting_board", True)' in op
        assert '"docs/nhl_v2_shadow.json"' in op
        assert '"docs/nhl_predictions.json"' not in op
        assert '"docs/nhl_picks_log.jsonl"' not in op
        assert "cancel-in-progress: false" in wf
        assert "github.ref == 'refs/heads/main'" in wf
        assert "failed_or_unrecorded_decisions" in op


if __name__=="__main__":
    unittest.main()
