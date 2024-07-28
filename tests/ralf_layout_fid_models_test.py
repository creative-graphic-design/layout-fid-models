from layout_fidnet_v3.configuration_layout_fidnet_v3 import LayoutFIDNetV3Config
from layout_fidnet_v3.modeling_layout_fidnet_v3 import (
    LayoutFIDNetV3,
    convert_from_checkpoint,
)


def test_load_ralf_cgl_max10(
    repo_id: str,
    fidnet_v3_d_model: int,
    fidnet_v3_nhead: int,
    fidnet_v3_num_layers: int,
    cgl_num_labels: int = 4,
    cgl_max_seq_len: int = 10,
    checkpoint_filename: str = "ralf_cgl_max10_model-best.pth.tar",
):
    config = LayoutFIDNetV3Config(
        num_labels=cgl_num_labels,
        d_model=fidnet_v3_d_model,
        nhead=fidnet_v3_nhead,
        num_layers=fidnet_v3_num_layers,
        max_bbox=cgl_max_seq_len,
    )
    model = convert_from_checkpoint(
        repo_id=repo_id,
        filename=checkpoint_filename,
        config=config,
    )
    assert isinstance(model, LayoutFIDNetV3)


def test_load_ralf_rico25_max25(
    repo_id: str,
    fidnet_v3_d_model: int,
    fidnet_v3_nhead: int,
    fidnet_v3_num_layers: int,
    pku_num_labels: int = 3,
    pku_max_seq_len: int = 10,
    checkpoint_filename: str = "ralf_pku10_max10_model-best.pth.tar",
):
    config = LayoutFIDNetV3Config(
        num_labels=pku_num_labels,
        d_model=fidnet_v3_d_model,
        nhead=fidnet_v3_nhead,
        num_layers=fidnet_v3_num_layers,
        max_bbox=pku_max_seq_len,
    )
    model = convert_from_checkpoint(
        repo_id=repo_id,
        filename=checkpoint_filename,
        config=config,
    )
    assert isinstance(model, LayoutFIDNetV3)
