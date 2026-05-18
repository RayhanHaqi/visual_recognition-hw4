from pathlib import Path


def test_autosave_commits_generated_files_before_rebase():
    script = Path("train_twostage.sh").read_text().splitlines()
    commands = [line.strip() for line in script]

    add_index = commands.index("git add -A")
    commit_index = next(
        i for i, line in enumerate(commands)
        if line.startswith("git commit -m ")
    )
    pull_index = commands.index("git pull --rebase")
    push_index = commands.index("git push")

    assert add_index < commit_index < pull_index < push_index
