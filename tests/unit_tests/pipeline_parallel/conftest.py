"""Keep SlackPipe's repeated-context tests within their process-group lifetime."""

import pytest


@pytest.fixture(autouse=True)
def slackpipe_process_group_ownership(request):
    if "slackpipe" not in request.node.nodeid:
        yield
        return
    from tests.unit_tests.pipeline_parallel.slackpipe_test_utils import (
        managed_model_parallel_groups,
    )

    with managed_model_parallel_groups():
        yield
