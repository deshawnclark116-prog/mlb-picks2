"""The documented research environment must be exactly what CI installs."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_research_requirements_match_every_v2_workflow_pin():
    req = (ROOT / 'requirements-research.txt').read_text()
    pin = re.search(r'^numpy==([\d.]+)$', req, re.M).group(1)
    for wf in sorted((ROOT / '.github/workflows').glob('nfl_v2_*.yml')):
        text = wf.read_text()
        assert 'python-version: "3.12"' in text, wf.name
        for m in re.finditer(r'numpy==([\d.]+)', text):
            assert m.group(1) == pin, wf.name


def test_setup_script_uses_the_ci_python_and_requirements():
    s = (ROOT / 'scripts/setup_research_env.sh').read_text()
    assert 'python3.12' in s and 'requirements-research.txt' in s and 'tests/test_nfl_v2_*.py' in s
