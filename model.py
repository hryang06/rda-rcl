import math
from turtle import forward

import torch
# import torch.utils.checkpoint
# from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss, CosineSimilarity, PairwiseDistance

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaClassificationHead, RobertaPooler
from transformers.modeling_outputs import SequenceClassifierOutput

class Similarity(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp # temperature hyperparameter
        self.cosSim = CosineSimilarity(dim=-1)
        
    def forward(self, x, y):
        return self.cosSim(x, y) / self.temp

class CL_LogSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.similarity = Similarity()
        self.loss_fct = CrossEntropyLoss()

    def forward(self, encoder_output, labels):
        num_info = encoder_output.size(1)
        logits = None
        for i in range(1, num_info):
            sim = self.similarity(encoder_output[:,0], encoder_output[:,i]) # (b,)
            sim = sim.unsqueeze(1) # (b,1)
            logits = sim if logits is None else torch.cat((logits, sim), dim=1) # (b,3)
        # pos_sim = self.similarity(base_output.unsqueeze(1), pos_output.unsqueeze(0)) # (b,b)
        # neg_sim = self.similarity(base_output.unsqueeze(1), neg_output.unsqueeze(0)) # (b,b)
        # logits = torch.cat([pos_sim, neg_sim], dim=1) # (b,2b)
        # logits = pos_sim
        # labels = torch.arange(logits.size(0)).long().to(base_output.device) # (b,)
        loss = self.loss_fct(logits, labels)
        
        return loss, logits # REMOVE!!!!

class CL_DistanceMaxLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, base_output, pos_output, neg_output):
        pos_dist = torch.sqrt(torch.sum(torch.pow(base_output - pos_output, 2), dim=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(base_output - neg_output, 2), dim=-1))
        logits = self.margin + torch.mean(pos_dist) - torch.mean(neg_dist)
        loss = torch.max(torch.tensor([0.0, logits])).to(base_output.device)
        
        return loss

def cl_loss(pos, neg, margin=1.0):
    logits = margin + torch.mean(pos) - torch.mean(neg)
    return torch.max(torch.tensor([0.0, logits]))

class BertForContrastiveLearning(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config, weight=0.1, is_weighted_sum=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.weight = weight if is_weighted_sum else -1.0
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
                
        self.similarity = Similarity()
        
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        is_cl_train = True if len(input_ids.size()) != 2 else False
        
        if is_cl_train: # (b,3,128)
            num_info = input_ids.size(1)
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (b*3, 128)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))
            
        # outputs for original classifier & contrastive learning
        outputs = self.bert(
            input_ids, # (b*3, 128)
            attention_mask=attention_mask, # (b*3, 128)
            token_type_ids=token_type_ids, # (b*3, 128)
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1] # (b*3,768)

        # only classification learning -----------------------------------------
        if not is_cl_train:
            pooled_output = self.dropout(pooled_output) # (b,768)
            logits = self.classifier(pooled_output) # (b,3)
        # only contrastive learning --------------------------------------------
        elif self.weight < 0.0:
            encoder_output = pooled_output.view((batch_size, num_info, pooled_output.size(-1))) # (b,3,768)
            pooled_output = encoder_output[:,0] # (b,768)
        # weighted sum (CLL & CEL) ---------------------------------------------
        else:
            org_labels = labels[:, 0]
            labels = labels.view((-1))
            
            # for classification
            encoder_output = pooled_output.view((batch_size, num_info, pooled_output.size(-1))) # (b,3,768)
            pooled_output = encoder_output[:,0] # (b,768)
            labels = org_labels
            
            pooled_output = self.dropout(pooled_output) # (b,768)
            logits = self.classifier(pooled_output) # (b,3)
        
        # Cross Entropy Loss
        loss = None
        if labels is not None:
            # only classification learning -----------------------------------------
            if not is_cl_train:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            # only contrastive learning --------------------------------------------
            elif self.weight < 0.0:
                # log similarity loss (i.e. SimCSE , pair-level)
                loss_cl = CL_LogSimilarityLoss()
                loss_c = loss_cl(encoder_output, labels)
                loss = loss_c[0]
                logits = loss_c[1]
            # weighted sum (CLL & CEL) ---------------------------------------------
            else:
                # log similarity loss (i.e. SimCSE , pair-level)
                loss_cl = CL_LogSimilarityLoss()
                loss_c = loss_cl(encoder_output, org_labels)
                
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                                    
                loss += self.weight * loss_c[0]
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            hidden_states=pooled_output, # encoder_output !!!
            attentions=outputs.attentions,
        )

class RobertaForContrastiveLearning(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, weight=0.1, is_weighted_sum=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.weight = weight if is_weighted_sum else 0.0
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.pooler = RobertaPooler(config)
        self.classifier = RobertaClassificationHead(config)
                
        self.similarity = Similarity()
        
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_length = input_ids.size(0), input_ids.size(-1)
        is_cl_train = True if len(input_ids.size()) != 2 else False
        if is_cl_train: # (b,3,128)
            num_info = input_ids.size(1)
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (b*3, 128)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
            
        # outputs for original classifier & contrastive learning
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, # True
        )
        pooled_output = outputs[0] # (b,128,768)
        
        if is_cl_train:
            pooled_output = self.pooler(pooled_output)
            encoder_output = pooled_output.view((batch_size, num_info, pooled_output.size(-1))) # (b,3,768)
            # pooled_output = encoder_output[:,0] # (b,768)
        else:
            logits = self.classifier(pooled_output) # (b,3)
        # logits = self.classifier(encoder_output) # (b,3)
            
        # Cross Entropy Loss
        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits, labels)
        
            # Contrastive Learning Loss
            if is_cl_train:
                # log similarity loss (i.e. SimCSE , pair-level ...)
                loss_cl = CL_LogSimilarityLoss()
                loss_c = loss_cl(encoder_output, labels)
                loss = loss_c[0]
                logits = loss_c[1]
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            hidden_states=pooled_output, # encoder_output !!!
            attentions=outputs.attentions,
        )
