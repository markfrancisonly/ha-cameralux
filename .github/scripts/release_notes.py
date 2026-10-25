"""Release notes from the commits since the previous tag.

One bullet per commit subject, the commit body's own bullets indented under
it. Version bumps, workflow updates, merges and production syncs are left out.
The first release of a repo just says so.
Usage: release_notes.py <tag> <owner/repo>   (run inside the repo)
"""
import re
import subprocess
import sys

SKIP = re.compile(r"^(Bump (manifest|version)\b|Update workflow actions\b|Merge |Sync production\b)")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def body_bullets(body):
    bullets = []
    for line in body.splitlines():
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif bullets and line[:1].isspace() and line.strip():
            bullets[-1] += " " + line.strip()
        elif not line.strip() and bullets:
            bullets.append(None)  # a blank line ends the current bullet
    return [b for b in bullets if b]


def notes(tag, repo):
    prev = git("describe", "--tags", "--abbrev=0", f"{tag}^").strip()
    if not prev:
        return "Initial release."
    lines = []
    for sha in git("log", "--reverse", "--no-merges", "--format=%H", f"{prev}..{tag}").split():
        subject = git("log", "-1", "--format=%s", sha).strip()
        if SKIP.match(subject):
            continue
        lines.append(f"- {subject}")
        lines += [f"  - {b}" for b in body_bullets(git("log", "-1", "--format=%b", sha))]
    lines.append(f"\n**Full Changelog**: https://github.com/{repo}/compare/{prev}...{tag}")
    return "\n".join(lines).strip()


if __name__ == "__main__":
    print(notes(sys.argv[1], sys.argv[2]))
