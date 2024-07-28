import pytest


@pytest.fixture
def repo_id() -> str:
    return "creative-graphic-design/layout-fid-checkpoints"


@pytest.fixture
def fidnet_v3_d_model() -> int:
    return 256


@pytest.fixture
def fidnet_v3_nhead() -> int:
    return 4


@pytest.fixture
def fidnet_v3_num_layers() -> int:
    return 4
