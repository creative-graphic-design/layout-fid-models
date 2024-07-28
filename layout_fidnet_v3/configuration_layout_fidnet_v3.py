from transformers.configuration_utils import PretrainedConfig


class LayoutFIDNetV3Config(PretrainedConfig):
    model_type = "layoutdm_fidnet_v3"

    def __init__(
        self,
        num_labels: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        max_bbox: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            **kwargs,
        )
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_bbox = max_bbox
