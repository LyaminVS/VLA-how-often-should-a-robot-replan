"""Bootstrap must not depend on pip/ensurepip in the notebook kernel Python."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/setup_action_dropout_kaggle.sh"


def executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def fake_program(body):
    return f"#!{sys.executable}\n" + textwrap.dedent(body)


@pytest.mark.parametrize("partial_tools", [False, True])
@pytest.mark.parametrize("inherited_prefix_vars", [
    (), ("UV_INSTALL_DIR",), ("CARGO_DIST_FORCE_INSTALL_DIR",),
    ("UV_INSTALL_DIR", "CARGO_DIST_FORCE_INSTALL_DIR"),
])
def test_setup_works_without_host_ensurepip_and_can_be_rerun(
        tmp_path, partial_tools, inherited_prefix_vars):
    """Exercise shell orchestration, not downloads or heavyweight package installs."""
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(SOURCE, root / "scripts/setup_action_dropout_kaggle.sh")
    (root / "requirements").mkdir()
    (root / "requirements/linux.lock.txt").write_text("# Fixture lock\n")
    stub_bin = tmp_path / "host-bin"
    stub_bin.mkdir()
    log = tmp_path / "calls.jsonl"
    sentinel = fake_program("""
        import json, os, sys
        with open(os.environ['BOOTSTRAP_CALL_LOG'], 'a') as stream:
            stream.write(json.dumps(['forbidden-host-python', *sys.argv[1:]]) + '\\n')
        raise SystemExit(97)
    """)
    executable(stub_bin / "python3", sentinel)
    executable(stub_bin / "python", sentinel)
    if partial_tools:
        # The original failing `venv .tools` leaves interpreters but no usable pip.
        executable(root / ".tools/bin/python3", sentinel)
        executable(root / ".tools/bin/python", sentinel)

    managed_python = fake_program("""
        import json, os, sys
        with open(os.environ['BOOTSTRAP_CALL_LOG'], 'a') as stream:
            stream.write(json.dumps(['managed-python', sys.argv[0], *sys.argv[1:]]) + '\\n')
        if sys.argv[1:2] == ['-']:
            sys.stdin.read()
        if sys.argv[1:2] == ['--version']:
            print('Python 3.12.3')
    """)
    python_source = tmp_path / "managed-python"
    executable(python_source, managed_python)
    uv_source = tmp_path / "fake-uv"
    executable(uv_source, fake_program("""
        import json, os, pathlib, shutil, sys
        args = sys.argv[1:]
        with open(os.environ['BOOTSTRAP_CALL_LOG'], 'a') as stream:
            stream.write(json.dumps(['uv', *args]) + '\\n')
        if args == ['--version']:
            print('uv 0.13.1')
        elif args and args[0] == 'venv':
            directory = pathlib.Path(args[-1]) / 'bin'
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(os.environ['FAKE_MANAGED_PYTHON'], directory / 'python')
            (directory / 'python').chmod(0o755)
        elif args[:2] not in (['pip', 'sync'], ['pip', 'install']):
            raise SystemExit('Unexpected uv command: ' + repr(args))
    """))
    installer = tmp_path / "uv-install.sh"
    installer.write_text("""#!/usr/bin/env bash
set -eu
# Mirror the pinned installer's precedence, not just our preferred variable.
destination="${UV_INSTALL_DIR:-${CARGO_DIST_FORCE_INSTALL_DIR:-${UV_UNMANAGED_INSTALL:-}}}"
test -n "$destination"
mkdir -p "$destination"
cp "$FAKE_UV_SOURCE" "$destination/uv"
chmod +x "$destination/uv"
""")
    executable(stub_bin / "curl", fake_program("""
        import json, os, pathlib, sys
        args = sys.argv[1:]
        with open(os.environ['BOOTSTRAP_CALL_LOG'], 'a') as stream:
            stream.write(json.dumps(['curl', *args]) + '\\n')
        content = pathlib.Path(os.environ['FAKE_UV_INSTALLER']).read_bytes()
        target = None
        for flag in ('-o', '--output'):
            if flag in args:
                target = args[args.index(flag) + 1]
        if target is None:
            sys.stdout.buffer.write(content)
        else:
            pathlib.Path(target).write_bytes(content)
    """))
    env = {**os.environ, "PATH": str(stub_bin) + os.pathsep + os.environ["PATH"],
           "BOOTSTRAP_CALL_LOG": str(log), "FAKE_UV_SOURCE": str(uv_source),
           "FAKE_UV_INSTALLER": str(installer), "FAKE_MANAGED_PYTHON": str(python_source)}
    # Notebook images may export installer prefixes pointing to /usr/local/bin.
    # Use owned temporary stand-ins and verify that the bootstrap neither uses
    # nor modifies those inherited locations, including on a second invocation.
    foreign_prefixes = []
    for variable in ("UV_INSTALL_DIR", "CARGO_DIST_FORCE_INSTALL_DIR",
                     "UV_UNMANAGED_INSTALL"):
        env.pop(variable, None)
    for variable in inherited_prefix_vars:
        foreign_prefix = tmp_path / f"outside-prefix-{variable}"
        foreign_prefix.mkdir()
        (foreign_prefix / "existing-tool").write_bytes(b"must remain unchanged\n")
        env[variable] = str(foreign_prefix)
        foreign_prefixes.append(foreign_prefix)
    for _ in range(2):
        result = subprocess.run(["bash", str(root / "scripts/setup_action_dropout_kaggle.sh")],
                                cwd=root, env=env, text=True, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        assert (root / ".tools/bin/uv").read_bytes() == uv_source.read_bytes()
        for foreign_prefix in foreign_prefixes:
            assert sorted(path.name for path in foreign_prefix.iterdir()) == ["existing-tool"]
            assert (foreign_prefix / "existing-tool").read_bytes() == b"must remain unchanged\n"

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(row[0] == "forbidden-host-python" for row in calls)
    downloads = [row for row in calls if row[0] == "curl"]
    assert len(downloads) == 1, "The installed standalone uv should be reusable"
    assert "https://astral.sh/uv/0.13.1/install.sh" in downloads[0]
    environments = [row for row in calls if row[:2] == ["uv", "venv"]]
    assert len(environments) == 2, "Create build tools and experiment environments only once"
    assert {Path(row[-1]).name for row in environments} == {".build-tools", ".venv"}
    for command in environments:
        assert command[command.index("--python") + 1] == "3.12.3"
        assert command[command.index("--python-preference") + 1] == "only-managed"
    installs = [row for row in calls if row[:3] == ["uv", "pip", "install"]]
    assert len(installs) == 2
    assert all("cmake==3.31.6" in row for row in installs)
    syncs = [row for row in calls if row[:3] == ["uv", "pip", "sync"]]
    assert len(syncs) == 2
    assert all(row[-1] == "requirements/linux.lock.txt" for row in syncs)
    python_calls = [row for row in calls if row[0] == "managed-python"]
    assert sum(row[-1] == "scripts/prepare_libero.py" for row in python_calls) == 2
    assert sum(row[-1] == "-" for row in python_calls) == 2
