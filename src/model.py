import config
import torch
import transformers
import torch.nn as nn
from torch.nn import functional as F


class TweetModel(transformers.RobertaModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_PATH, config=conf
        )
        self.l0 = nn.Linear(768 * 2, 2)
        nn.init.normal_(self.l0.weight, std=0.02)

        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_p) for dropout_p in np.linspace(0.3, 0.6, 3)]
        )
        self.seq = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(768 * 3, 768 * 2), nn.ReLU(inplace=True),
        )

    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        # hidden_layers = [layers, batches, tokens len, features]
        #                 [13,        32,          160,       768]

        out = torch.cat((out[-1], out[-2], out[-3]), dim=-1)  # [32, 160, 2304]

        seq_res = self.seq(out)
        start_logits, end_logits = torch.mean(
            torch.stack(
                [
                    # Multi-Sample Dropout takes place here
                    self.l0(dropout(seq_res))
                    for dropout in self.dropouts
                ],
                dim=0,
            ),
            dim=0,
        ).split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
