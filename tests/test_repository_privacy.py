"""Keep deployment-specific identifiers out of public source and examples."""

from pathlib import Path
import subprocess


def test_no_private_deployment_literals():
    root = Path(__file__).resolve().parents[1]
    # Construct the denylist so this regression check does not itself contain the literals.
    forbidden = (
        "markosnarinian" + ".com", "192.168." + "1.", "194.154." + "34.146",
        "/home/" + "markos", "wd" + "2tb", "elite" + "book",
        "enp0s" + "31f6", "en" + "o2",
    )
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
    ).split(b"\0")
    failures = []
    for name in set(paths) - {b""}:
        path = root / name.decode()
        if not path.is_file():
            continue
        # Read binary files too: a literal leak in a fixture is still a leak.
        data = path.read_bytes().replace(b"github.com/markosnarinian/YOLOX", b"PUBLIC_REPOSITORY")
        for value in forbidden:
            if value.encode() in data:
                failures.append(f"{path.relative_to(root)}: private deployment identifier")
    assert not failures, "\n".join(failures)
