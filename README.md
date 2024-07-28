# Layout FID models

[![CI](https://github.com/creative-graphic-design/layout-fid-models/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/layout-fid-models/actions/workflows/ci.yaml)

```python
from transformers import AutoModel

repo_id = "creative-graphic-design/layout-fidnet-v3-layoutdm-publaynet"
fid_model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

print(fid_model)
# LayoutFIDNetV3(
#   (emb_label): Embedding(5, 256)
#   (fc_bbox): Linear(in_features=4, out_features=256, bias=True)
#   (enc_fc_in): Linear(in_features=512, out_features=256, bias=True)
#   (enc_transformer): TransformerWithToken(
#     (core): TransformerEncoder(
#       (layers): ModuleList(
#         (0-3): 4 x TransformerEncoderLayer(
#           (self_attn): MultiheadAttention(
#             (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
#           )
#           (linear1): Linear(in_features=256, out_features=128, bias=True)
#           (dropout): Dropout(p=0.1, inplace=False)
#           (linear2): Linear(in_features=128, out_features=256, bias=True)
#           (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#           (dropout1): Dropout(p=0.1, inplace=False)
#           (dropout2): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#   )
#   (fc_out_disc): Linear(in_features=256, out_features=1, bias=True)
#   (dec_fc_in): Linear(in_features=512, out_features=256, bias=True)
#   (dec_transformer): TransformerEncoder(
#     (layers): ModuleList(
#       (0-3): 4 x TransformerEncoderLayer(
#         (self_attn): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
#         )
#         (linear1): Linear(in_features=256, out_features=128, bias=True)
#         (dropout): Dropout(p=0.1, inplace=False)
#         (linear2): Linear(in_features=128, out_features=256, bias=True)
#         (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#         (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#         (dropout1): Dropout(p=0.1, inplace=False)
#         (dropout2): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
#   (fc_out_cls): Linear(in_features=256, out_features=5, bias=True)
#   (fc_out_bbox): Linear(in_features=256, out_features=4, bias=True)
# )
```
