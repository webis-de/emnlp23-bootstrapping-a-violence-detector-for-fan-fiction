# import torch
# from transformers import BertModel
# import torch.nn.functional as F
# import numpy as np
# from typing import List, Optional, Tuple, Union
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModelForSequenceClassification


def get_model(checkpoint, num_labels, model_type):
    """
    :param checkpoint:
    :param num_labels:
    :param model_type: `longformer` or `tobert` or `bert`, depending on the wanted model
    """
    if model_type == 'longformer':
        return AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to('cuda')
    # elif model_type == 'tobert':
    #     return ToBERTModel(num_labels, 'cuda')
    elif model_type == 'bert':
        return AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to('cuda')
    else:
        raise TypeError(f"model_type must be longformer|tobert|bert but was {model_type}")

#
# class ToBERTModel(torch.nn.Module):
#     """ This model is from https://github.com/amazon-science/efficient-longdoc-classification/blob/main/src/models.py
#      """
#     def __init__(self, num_labels, device):
#         super(ToBERTModel, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.trans = torch.nn.TransformerEncoderLayer(d_model=768, nhead=2)
#         self.fc = torch.nn.Linear(768, 30)
#         self.classifier = torch.nn.Linear(30, num_labels)
#         self.device = device
#
#     def forward(
#             self,
#             input_ids: Optional[torch.Tensor] = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             token_type_ids: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.Tensor] = None,
#             head_mask: Optional[torch.Tensor] = None,
#             inputs_embeds: Optional[torch.Tensor] = None,
#             labels: Optional[torch.Tensor] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#
#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask,
#                             inputs_embeds=inputs_embeds,
#                             output_attentions=output_attentions,
#                             output_hidden_states=output_hidden_states,
#                             return_dict=False)
#
#         pooled_out = outputs[1]
#         chunks_emb = pooled_out.split_with_sizes(length)
#         batch_emb_pad = torch.nn.utils.rnn.pad_sequence(
#             chunks_emb, padding_value=0, batch_first=True)
#         batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
#         padding_mask = np.zeros([batch_emb.shape[1], batch_emb.shape[0]]) # Batch size, Sequence length
#         for idx in range(len(padding_mask)):
#             padding_mask[idx][length[idx]:] = 1 # padding key = 1 ignored
#
#         padding_mask = torch.tensor(padding_mask).to(self.device, dtype=torch.bool)
#         trans_output = self.trans(batch_emb, src_key_padding_mask=padding_mask)
#         mean_pool = torch.mean(trans_output, dim=0) # Batch size, 768
#         fc_output = self.fc(mean_pool)
#         relu_output = F.relu(fc_output)
#         logits = self.classifier(relu_output)
#
#         return tuple(logits)
#
#         # return SequenceClassifierOutput(
#         #     loss=loss,
#         #     logits=logits,
#         #     hidden_states=outputs.hidden_states,
#         #     attentions=outputs.attentions,
#         # )