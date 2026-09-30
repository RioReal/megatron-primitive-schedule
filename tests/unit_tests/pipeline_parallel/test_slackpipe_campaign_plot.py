# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resume avoids reparsing completed captures, but repairs interrupted plots."""

import json

import pytest

from tools import plot_schedule_trace
from tools.run_slackpipe_real_system_campaign import plot_campaign
from tools.slackpipe_eval_config import fingerprint


@pytest.mark.parametrize("complete", [True, False])
def test_campaign_plot_completion(tmp_path, monkeypatch, complete):
    receipts = {
        s: dict(status="passed", directory=f"traces/{s}") for s in ("interleaved", "slackpipe")
    }
    destination = (
        tmp_path
        / "traces"
        / ("figure-" + fingerprint({s: r["directory"] for s, r in receipts.items()})[:12])
    )
    destination.mkdir(parents=True)
    (destination / "timeline.png").write_bytes(b"old interrupted PNG")
    if complete:
        (destination / "timeline.pdf").write_bytes(b"PDF")
        (destination / "figure_report.json").write_text('{"selection": {"iteration": 23}}')
    parsed = []

    def iterations(path, cycle):
        assert not complete, "A completed figure must be skipped before trace parsing"
        parsed.append(path)
        assert cycle == 0
        return {23, 24} if path.name == "interleaved" else {24, 25}

    def panel(path, iteration, cycle):
        assert iteration == 24 and cycle == 0
        return path

    def render(*args, png, pdf, **kwargs):
        png.write_bytes(b"new PNG")
        pdf.write_bytes(b"new PDF")
        return dict(panels=2)

    monkeypatch.setattr(plot_schedule_trace, "available_iterations", iterations)
    monkeypatch.setattr(plot_schedule_trace, "load_panel", panel)
    monkeypatch.setattr(plot_schedule_trace, "render", render)
    plot_campaign(tmp_path, receipts)
    assert len(parsed) == (0 if complete else 2)
    if not complete:
        report = json.loads((destination / "figure_report.json").read_text())
        assert report["selection"]["iteration"] == 24
        assert report["selection"]["candidates"] == [24]
        assert (destination / "timeline.png").read_bytes() == b"new PNG"
