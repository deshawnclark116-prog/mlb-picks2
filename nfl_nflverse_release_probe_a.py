"""
One-off diagnostic: list nflverse-data release tags and their assets so we
can find where (if anywhere) 2025 NFL season player-week stats actually
live. The 'player_stats' release tag's asset list has been confirmed empty
of anything containing '2025' across three separate resolver attempts --
this checks whether the data exists under a different release tag entirely
before concluding it isn't published yet.
"""
import json
import urllib.request

RELEASES_LIST_API = "https://api.github.com/repos/nflverse/nflverse-data/releases?per_page=100&page={page}"


def _http_get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-release-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    all_releases = []
    page = 1
    while True:
        batch = _http_get_json(RELEASES_LIST_API.format(page=page))
        if not batch:
            break
        all_releases.extend(batch)
        page += 1
        if page > 10:
            break

    print(f"total releases found: {len(all_releases)}")
    print()

    candidates = []
    for rel in all_releases:
        tag = rel.get("tag_name", "")
        if "player" in tag.lower() or "stat" in tag.lower() or "2025" in tag:
            candidates.append(rel)

    print(f"releases with 'player'/'stat'/'2025' in tag name: {len(candidates)}")
    for rel in candidates:
        tag = rel.get("tag_name", "")
        assets = rel.get("assets", [])
        names_2025 = [a["name"] for a in assets if "2025" in a["name"]]
        print(f"  tag={tag!r}  published={rel.get('published_at')}  n_assets={len(assets)}  2025_assets={names_2025[:5]}")

    print()
    print("all tag names containing '2025':")
    for rel in all_releases:
        tag = rel.get("tag_name", "")
        if "2025" in tag:
            print(f"  {tag}  published={rel.get('published_at')}")


if __name__ == "__main__":
    main()
