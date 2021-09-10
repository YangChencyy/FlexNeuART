#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import torch

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart.models.utils import init_model
from flexneuart.models import register
from flexneuart.models.base_bert_split_slide_window import \
        BertSplitSlideWindowRanker, DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE, \
        CLS_AGGREG_STACK
from flexneuart.models.base_bert import DEFAULT_BERT_DROPOUT

class Empty:
    pass

@register('parade_transformer')
class ParadeMaxRanker(BertSplitSlideWindowRanker):
    """
        PARADE Max ranker. Contributed by Tianyi Lin

        Li, C., Yates, A., MacAvaney, S., He, B., & Sun, Y. (2020). PARADE:
        Passage representation aggregation for document reranking.
        arXiv preprint arXiv:2008.09093.
    """
    def __init__(self, bert_flavor, bert_aggreg_flavor,
                 window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                 dropout=DEFAULT_BERT_DROPOUT):
        super().__init__(bert_flavor, cls_aggreg_type=CLS_AGGREG_STACK,
                         window_size=window_size, stride=stride)
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout)

        # Let's create an aggregator BERT
        init_data = Empty()
        init_model(init_data, bert_aggreg_flavor)
        # Must memorize this as a class attribute
        self.bert_aggreg = init_data.bert

        self.BERT_AGGREG_SIZE = init_data.BERT_SIZE
        self.bert_aggreg_cls_embed = torch.nn.Parameter(torch.FloatTensor(self.BERT_AGGREG_SIZE))
        # If there's a mismatch between the embedding size of the aggregating BERT and the
        # hidden size of the main BERT, a projection is required
        if self.BERT_SIZE != self.BERT_AGGREG_SIZE:
            self.proj_out = torch.nn.Linear(self.BERT_SIZE, self.BERT_AGGREG_SIZE)
            torch.nn.init.xavier_uniform_(self.proj_out.weight)
        else:
            self.proj_out = None

        self.cls = torch.nn.Linear(self.BERT_AGGREG_SIZE, 1)
        torch.nn.init.xavier_uniform_(self.cls.weight)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask):
        cls_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        last_layer_cls_rep = torch.transpose(cls_reps[-1], 1, 2)  # [B, N, BERT_SIZE]

        B, N, _ = last_layer_cls_rep.shape

        if self.proj_out is not None:
            last_layer_cls_rep_proj = self.proj_out(last_layer_cls_rep.view(B * N, -1)).view(B, N, -1) # [B, N, BERT_AGGREG_SIZE]
        else:
            last_layer_cls_rep_proj = last_layer_cls_rep

        # We need to prepend a CLS token vector as the classifier operation depends on the existence of such special token!
        last_layer_cls_rep_proj = torch.cat(last_layer_cls_rep_proj,
                                        # +two singletown dimensions before the CLS embedding
                                        self.bert_aggreg_cls_embed.unsqueeze(dim=0).unsqueeze(dim=0), # [1, 1, BERT_AGGREG_SIZE]
                                        dim=1) #[B, N+1, BERT_AGGREG_SIZE]

        # run aggregating BERT and get the last layer output
        # note that we pass directly vectors (CLS vector including!) without carrying out an embedding, b/c
        # it's pointless at this stage
        outputs : BaseModelOutputWithPoolingAndCrossAttentions = self.bert_aggreg(input_embeds=last_layer_cls_rep_proj)
        result = outputs.last_hidden_state

        # The cls vector of the last Transformer output layer
        parade_cls_reps = result[:, 0, :] #

        out = self.cls(self.dropout(parade_cls_reps))
        # the last dimension is singleton and needs to be removed
        return out.squeeze(dim=-1)