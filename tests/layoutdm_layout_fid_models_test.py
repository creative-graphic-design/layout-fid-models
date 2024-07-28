from layout_fidnet_v3.configuration_layout_fidnet_v3 import LayoutFIDNetV3Config
from layout_fidnet_v3.modeling_layout_fidnet_v3 import (
    LayoutFIDNetV3,
    convert_from_checkpoint,
)


def test_load_layoutdm_publaynet_max25(
    repo_id: str,
    fidnet_v3_d_model: int,
    fidnet_v3_nhead: int,
    fidnet_v3_num_layers: int,
    publaynet_num_labels: int = 5,
    publaynet_max_seq_len: int = 25,
    checkpoint_filename: str = "layoutdm_publaynet_max25_model-best.pth.tar",
):
    config = LayoutFIDNetV3Config(
        num_labels=publaynet_num_labels,
        d_model=fidnet_v3_d_model,
        nhead=fidnet_v3_nhead,
        num_layers=fidnet_v3_num_layers,
        max_bbox=publaynet_max_seq_len,
    )
    model = convert_from_checkpoint(
        repo_id=repo_id,
        filename=checkpoint_filename,
        config=config,
    )
    assert isinstance(model, LayoutFIDNetV3)


def test_load_layoutdm_rico25_max25(
    repo_id: str,
    fidnet_v3_d_model: int,
    fidnet_v3_nhead: int,
    fidnet_v3_num_layers: int,
    rico_num_labels: int = 25,
    rico_max_seq_len: int = 25,
    checkpoint_filename: str = "layoutdm_rico25_max25_model-best.pth.tar",
):
    config = LayoutFIDNetV3Config(
        num_labels=rico_num_labels,
        d_model=fidnet_v3_d_model,
        nhead=fidnet_v3_nhead,
        num_layers=fidnet_v3_num_layers,
        max_bbox=rico_max_seq_len,
    )
    model = convert_from_checkpoint(
        repo_id=repo_id,
        filename=checkpoint_filename,
        config=config,
    )
    assert isinstance(model, LayoutFIDNetV3)
