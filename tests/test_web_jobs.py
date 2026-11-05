import asyncio
import json

import pytest

from failure_atlas import web


class FakeStdout:
    def __init__(self, lines: list[str], fail_after: int | None = None) -> None:
        self.lines = lines
        self.fail_after = fail_after

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for index, line in enumerate(self.lines):
            if self.fail_after is not None and index == self.fail_after:
                raise ValueError("chunk exceeded the limit")
            yield (line + "\n").encode()


class FakeProcess:
    def __init__(self, lines, return_code=0, fail_after=None) -> None:
        self.stdout = FakeStdout(lines, fail_after)
        self._return_code = return_code
        self.returncode = None
        self.killed = False

    async def wait(self) -> int:
        self.returncode = self._return_code
        return self._return_code

    def kill(self) -> None:
        self.killed = True


@pytest.fixture(autouse=True)
def reset_job_state():
    saved = dict(web.job_state)
    yield
    web.job_state.clear()
    web.job_state.update(saved)


def patch_spawn(monkeypatch, process=None, error=None):
    async def spawn(*command, **options):
        assert options["limit"] == web.STREAM_LIMIT
        if error is not None:
            raise error
        return process

    monkeypatch.setattr(web.asyncio, "create_subprocess_exec", spawn)


def test_launch_failure_ends_the_job_as_error_instead_of_running_forever(monkeypatch) -> None:
    patch_spawn(monkeypatch, error=FileNotFoundError("python not found"))
    asyncio.run(web.run_commands("repair", ["scripts.repair_model"]))
    assert web.job_state["status"] == "error"
    assert web.job_state["return_code"] == -1
    assert web.job_state["finished_at"] is not None
    assert "FileNotFoundError" in web.job_state["logs"][-1]


def test_stream_failure_kills_the_process_and_reports_error(monkeypatch) -> None:
    process = FakeProcess(["step 1", "step 2", "step 3"], fail_after=1)
    patch_spawn(monkeypatch, process=process)
    asyncio.run(web.run_commands("rescan", ["scripts.search_failures"]))
    assert process.returncode is None  # never reached wait(); it was killed instead
    assert process.killed
    assert web.job_state["status"] == "error"


def test_failed_command_stops_the_chain_with_its_return_code(monkeypatch) -> None:
    patch_spawn(monkeypatch, process=FakeProcess(["boom"], return_code=3))
    asyncio.run(web.run_commands("repair", ["scripts.a", "scripts.b"]))
    assert web.job_state["status"] == "error"
    assert web.job_state["return_code"] == 3
    assert web.job_state["logs"].count("$ python -m scripts.b") == 0


def test_experiment_saves_the_full_log_even_when_the_run_id_scrolled_out(
    monkeypatch, tmp_path
) -> None:
    run_dir = tmp_path / "my-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps({"artifacts": []}), encoding="utf-8")
    lines = [f"[run]  workspace=C:\\data\\runs\\my-run"] + [f"epoch {i}" for i in range(300)]
    patch_spawn(monkeypatch, process=FakeProcess(lines, return_code=1))
    monkeypatch.setattr(web, "RUN_ROOT", tmp_path)
    request = web.ExperimentRequest(model="fashion_cnn", dataset="fashion_mnist")
    asyncio.run(web.run_agent_experiment(request))

    # The 120-line window no longer shows the workspace line, but the run id was kept.
    assert not any("workspace=" in line for line in web.job_state["logs"])
    assert web.job_state["run_id"] == "my-run"
    assert web.job_state["status"] == "error"
    saved = (run_dir / "execution.log").read_text(encoding="utf-8").splitlines()
    assert any("workspace=" in line for line in saved)
    assert saved[-1] == "epoch 299"
    assert json.loads((run_dir / "metadata.json").read_text())["artifacts"] == ["execution.log"]


def test_experiment_launch_failure_is_reported(monkeypatch) -> None:
    patch_spawn(monkeypatch, error=OSError("cannot start"))
    request = web.ExperimentRequest(model="fashion_cnn", dataset="fashion_mnist")
    asyncio.run(web.run_agent_experiment(request))
    assert web.job_state["status"] == "error"
    assert web.job_state["return_code"] == -1
    assert web.job_state["action"] == "experiment"


def test_experiment_command_carries_every_option() -> None:
    request = web.ExperimentRequest(
        model="lenet",
        dataset="mnist",
        stage="full",
        run_id="r1",
        epochs=2,
        seed=7,
        batch_size=64,
        perturbations=["rotation", "blur"],
        severity_scale=0.5,
        reuse_artifacts=False,
    )
    command = web.build_experiment_command(request)
    assert command[command.index("--model") + 1] == "lenet"
    assert command[command.index("--run-id") + 1] == "r1"
    assert command[command.index("--baseline-epochs") + 1] == "2"
    assert command[command.index("--weight-mode") + 1] == "scratch"
    assert command.count("--perturbation") == 2
    assert "--force" in command
