from cfb_team_boxscore_audit import parse_team_stats, reconcile

def block(passing="20-35", rushing="32"):
    return {"team":{"id":"1","displayName":"Example"}, "statistics":[
        {"name":"completionAttempts","displayValue":passing},
        {"name":"rushingAttempts","displayValue":rushing}]}

def test_team_totals():
    assert parse_team_stats(block()) == {"pass_attempts":35,"carries":32,"completions":20}

def test_missing_and_invalid():
    assert parse_team_stats(block("bad")) is None
    assert parse_team_stats(block("40-35")) is None
    assert parse_team_stats(block("0-0","0")) is None

def test_roster_undercount():
    rows = reconcile({"boxscore":{"teams":[block()]}},{"Example":{"pass_attempts":0,"carries":30}})
    assert rows[0]["status"] == "mismatch"
    assert rows[0]["official"]["pass_attempts"] == 35

def test_missing_player_rows():
    assert reconcile({"boxscore":{"teams":[block()]}},{})[0]["status"] == "missing_player_rows"
