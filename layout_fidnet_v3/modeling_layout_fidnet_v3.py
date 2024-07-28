import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_layout_fidnet_v3 import LayoutFIDNetV3Config

logger = logging.getLogger(__name__)


@dataclass
class LayoutFIDNetV3Output(ModelOutput):
    logit_disc: torch.Tensor
    logit_cls: torch.Tensor
    bbox_pred: torch.Tensor


class TransformerWithToken(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer("token_mask", token_mask)

        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask):
        # x: [N, B, E]
        # padding_mask: [B, N]
        #   `False` for valid values
        #   `True` for padded values

        B = x.size(1)

        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)

        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)

        x = self.core(x, src_key_padding_mask=padding_mask)

        return x


class LayoutFIDNetV3(PreTrainedModel):
    config_class = LayoutFIDNetV3Config

    def __init__(self, config: LayoutFIDNetV3Config) -> None:
        super().__init__(config)

        # encoder
        self.emb_label = nn.Embedding(config.num_labels, config.d_model)
        self.fc_bbox = nn.Linear(4, config.d_model)
        self.enc_fc_in = nn.Linear(config.d_model * 2, config.d_model)

        self.enc_transformer = TransformerWithToken(
            d_model=config.d_model,
            dim_feedforward=config.d_model // 2,
            nhead=config.nhead,
            num_layers=config.num_layers,
        )

        self.fc_out_disc = nn.Linear(config.d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(config.max_bbox, 1, config.d_model))
        self.dec_fc_in = nn.Linear(config.d_model * 2, config.d_model)

        te = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model // 2,
        )
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=config.num_layers)

        self.fc_out_cls = nn.Linear(config.d_model, config.num_labels)
        self.fc_out_bbox = nn.Linear(config.d_model, 4)

    def extract_features(
        self, bbox: torch.Tensor, label: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        b = self.fc_bbox(bbox)
        l = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0]

    def forward(
        self,
        bbox: torch.Tensor,
        label: torch.Tensor,
        padding_mask: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LayoutFIDNetV3Output]:
        B, N, _ = bbox.size()
        x = self.extract_features(bbox, label, padding_mask)

        logit_disc = self.fc_out_disc(x).squeeze(-1)

        x = x.unsqueeze(0).expand(N, -1, -1)
        t = self.pos_token[:N].expand(-1, B, -1)
        x = torch.cat([x, t], dim=-1)
        x = torch.relu(self.dec_fc_in(x))

        x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
        # x = x.permute(1, 0, 2)[~padding_mask]
        x = x.permute(1, 0, 2)

        # logit_cls: [B, N, L]    bbox_pred: [B, N, 4]
        logit_cls = self.fc_out_cls(x)
        bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

        if not return_dict:
            return logit_disc, logit_cls, bbox_pred

        return LayoutFIDNetV3Output(
            logit_disc=logit_disc, logit_cls=logit_cls, bbox_pred=bbox_pred
        )


def convert_from_checkpoint(
    repo_id: str, filename: str, config: Optional[LayoutFIDNetV3Config] = None
) -> LayoutFIDNetV3:
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    config = config or LayoutFIDNetV3Config()
    model = LayoutFIDNetV3(config)

    logger.info(f"Loading model from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model
