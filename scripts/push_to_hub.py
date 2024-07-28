import logging

from layout_fidnet_v3 import LayoutFIDNetV3Config, convert_from_checkpoint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def push_layout_fidnet_v3_to_hub(
    repo_id: str,
    push_to_repo_name: str,
    checkpoint_filename: str,
    num_labels: int,
    max_seq_len: int,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
):
    config = LayoutFIDNetV3Config(
        num_labels=num_labels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_bbox=max_seq_len,
    )
    model = convert_from_checkpoint(
        repo_id=repo_id,
        filename=checkpoint_filename,
        config=config,
    )

    config.register_for_auto_class()
    model.register_for_auto_class()

    model.push_to_hub(push_to_repo_name, private=True)  # type: ignore


def main():
    push_layout_fidnet_v3_to_hub(
        repo_id="creative-graphic-design/layout-fid-checkpoints",
        checkpoint_filename="layoutdm_publaynet_max25_model-best.pth.tar",
        push_to_repo_name="creative-graphic-design/layout-fidnet-v3-layoutdm-publaynet",
        num_labels=5,
        max_seq_len=25,
    )
    push_layout_fidnet_v3_to_hub(
        repo_id="creative-graphic-design/layout-fid-checkpoints",
        checkpoint_filename="layoutdm_rico25_max25_model-best.pth.tar",
        push_to_repo_name="creative-graphic-design/layout-fidnet-v3-layoutdm-rico25",
        num_labels=25,
        max_seq_len=25,
    )
    push_layout_fidnet_v3_to_hub(
        repo_id="creative-graphic-design/layout-fid-checkpoints",
        checkpoint_filename="ralf_cgl_max10_model-best.pth.tar",
        push_to_repo_name="creative-graphic-design/layout-fidnet-v3-ralf-cgl",
        num_labels=4,
        max_seq_len=10,
    )
    push_layout_fidnet_v3_to_hub(
        repo_id="creative-graphic-design/layout-fid-checkpoints",
        checkpoint_filename="ralf_pku10_max10_model-best.pth.tar",
        push_to_repo_name="creative-graphic-design/layout-fidnet-v3-ralf-pku10",
        num_labels=3,
        max_seq_len=10,
    )


if __name__ == "__main__":
    main()
