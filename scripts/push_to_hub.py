import logging
from typing import Dict, List

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
    id2label: Dict[int, str],
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
):
    label2id = {label: idx for idx, label in id2label.items()}
    config = LayoutFIDNetV3Config(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
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


def labels_to_id2label(labels: List[str]) -> Dict[int, str]:
    return {i: label for i, label in enumerate(labels)}


def main():
    push_layout_fidnet_v3_to_hub(
        repo_id="creative-graphic-design/layout-fid-checkpoints",
        checkpoint_filename="layoutdm_publaynet_max25_model-best.pth.tar",
        push_to_repo_name="creative-graphic-design/layout-fidnet-v3-layoutdm-publaynet",
        num_labels=5,
        max_seq_len=25,
        id2label=labels_to_id2label(
            labels=["text", "title", "list", "table", "figure"]
        ),
    )
    push_layout_fidnet_v3_to_hub(
        repo_id="creative-graphic-design/layout-fid-checkpoints",
        checkpoint_filename="layoutdm_rico25_max25_model-best.pth.tar",
        push_to_repo_name="creative-graphic-design/layout-fidnet-v3-layoutdm-rico25",
        num_labels=25,
        max_seq_len=25,
        id2label=labels_to_id2label(
            labels=[
                "Text",
                "Image",
                "Icon",
                "Text Button",
                "List Item",
                "Input",
                "Background Image",
                "Card",
                "Web View",
                "Radio Button",
                "Drawer",
                "Checkbox",
                "Advertisement",
                "Modal",
                "Pager Indicator",
                "Slider",
                "On/Off Switch",
                "Button Bar",
                "Toolbar",
                "Number Stepper",
                "Multi-Tab",
                "Date Picker",
                "Map View",
                "Video",
                "Bottom Navigation",
            ]
        ),
    )
    push_layout_fidnet_v3_to_hub(
        repo_id="creative-graphic-design/layout-fid-checkpoints",
        checkpoint_filename="ralf_cgl_max10_model-best.pth.tar",
        push_to_repo_name="creative-graphic-design/layout-fidnet-v3-ralf-cgl",
        num_labels=4,
        max_seq_len=10,
        id2label=labels_to_id2label(
            labels=[
                "logo",
                "text",
                "underlay",
                "embellishment",
            ]
        ),
    )
    push_layout_fidnet_v3_to_hub(
        repo_id="creative-graphic-design/layout-fid-checkpoints",
        checkpoint_filename="ralf_pku10_max10_model-best.pth.tar",
        push_to_repo_name="creative-graphic-design/layout-fidnet-v3-ralf-pku10",
        num_labels=3,
        max_seq_len=10,
        id2label=labels_to_id2label(
            labels=[
                "text",
                "logo",
                "underlay",
            ]
        ),
    )


if __name__ == "__main__":
    main()
