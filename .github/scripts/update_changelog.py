"""Auto-update CHANGELOG.md from conventional commits."""
import subprocess
import re
import os
from datetime import date

CHANGELOG = "CHANGELOG.md"
TODAY = date.today().isoformat()

def get_log():
    result = subprocess.run(
        ["git", "log", "origin/main..HEAD", "--pretty=format:%s"],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        result = subprocess.run(
            ["git", "log", "--pretty=format:%s", "-20"],
            capture_output=True, text=True
        )
    return result.stdout.strip().splitlines()

def categorize(messages):
    cats = {"Added": [], "Fixed": [], "Changed": [], "Other": []}
    for msg in messages:
        m = msg.strip()
        if not m or "[skip ci]" in m:
            continue
        if re.match(r"^feat", m, re.I):
            cats["Added"].append(m)
        elif re.match(r"^fix", m, re.I):
            cats["Fixed"].append(m)
        elif re.match(r"^(refactor|perf|style|docs|chore|build|ci)", m, re.I):
            cats["Changed"].append(m)
        else:
            cats["Other"].append(m)
    return cats

def build_block(cats):
    lines = [f"## [Unreleased] - {TODAY}", ""]
    for section, items in cats.items():
        if items:
            lines.append(f"### {section}")
            lines.append("")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")
    return "\n".join(lines)

def update():
    msgs = get_log()
    if not msgs:
        return
    cats = categorize(msgs)
    if not any(cats.values()):
        return
    block = build_block(cats)
    if not os.path.exists(CHANGELOG):
        content = "# Changelog\n\nAll notable changes documented here.\n\n" + block
    else:
        with open(CHANGELOG) as f:
            content = f.read()
        content = re.sub(r"## \[Unreleased\].*?(?=## \[|\Z)", "", content, flags=re.DOTALL)
        content = re.sub(r"(# Changelog.*?\n\n)", r"\1" + block + "\n\n", content, flags=re.DOTALL)
    with open(CHANGELOG, "w") as f:
        f.write(content)
    print(f"CHANGELOG.md updated: {sum(len(v) for v in cats.values())} entries")

if __name__ == "__main__":
    update()
