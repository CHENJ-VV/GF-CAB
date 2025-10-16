import numpy as np
from datasets import load_from_disk
import torch
from transformers import BertForMaskedLM
import os
import sys
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
# sys.path.append('/Users/chenj0i/Desktop/Lab Work/Geneformer')
from geneformer.pretrainer import token_dictionary
import datetime
import time
import pickle
import random
import subprocess
import numpy as np
import pytz
import torch
from datasets import load_from_disk, Dataset
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, TrainerCallback, Trainer, BertModel, BertPreTrainedModel
from geneformer import GeneformerPretrainer
from typing import Tuple
from torch import Tensor
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertOnlyMLMHead, BertPredictionHeadTransform
from transformers.activations import ACT2FN
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F

class CustomBertForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    _tied_weights_keys = ["decoder.weight", "bert.embeddings.word_embeddings.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.transform = BertPredictionHeadTransform(config)

        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

        # Initialize weights
        self.init_weights()

        # Tie weights automatically
        self.tie_weights()

        # self.post_init()

    def tie_weights(self):
        """
        Ties the weights between the input embeddings and output decoder weights.
        """
        self.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def probability_convert(self, probs: Tensor, input_ids: Tensor, labels: Tensor) -> Tensor:
        device = probs.device
        batch_size, seq_length, vocab_size = probs.size()
        _, input_seq_length = input_ids.size()

        # truncated_labels = labels[:, :input_seq_length]
        # non_mask = truncated_labels == -100
        non_mask = labels == -100
        non_mask_indices = non_mask.nonzero(as_tuple=True)        
        known_gene_indices = input_ids[non_mask]

        # Generate (1-p) matrix whiel assigning all known genes in the beginning
        zeros = torch.zeros((batch_size, 1, vocab_size), device=device)
        zeros[non_mask_indices[0], 0, known_gene_indices] = 1.0
        probs_shifted = torch.cat((zeros, probs[:, :-1, :]), dim=1)
        inv_probs_shifted = 1 - probs_shifted
        
        # Cumulative product to get (1-p_1)*(1-p_2)*...*(p_i)
        cumprod_inv_probs = torch.cumprod(inv_probs_shifted, dim=1)
        modified_probs = probs * cumprod_inv_probs

        # # Since we are assigning probabilities for already known genes, 
        # # (1-p_1)*(1-p_2)*...*(p_i) for these genes can result in 0, due to hard assignment of probs to be 1
        # # Add 1e-18 to avoid dividing modified probs by 0
        # # During dubugging stage, some issues occurred in the normalization step.
        # # Since probabilities in each position do not necessarily need to sum up to one, leave out normalization.
        normalized_probs = modified_probs.sum(dim=-1, keepdim=True).clamp(min=1e-18)
        modified_probs = modified_probs / normalized_probs # Normalization after cumulative production
        
        return modified_probs
    
    def assign_known_gene_probs(self, probs: Tensor, input_ids: Tensor, labels: Tensor) -> Tensor:

        device = probs.device
        batch_size, seq_length, vocab_size = probs.size()
        _, input_seq_length = input_ids.size()

        # Truncate `labels` to match the length of `input_ids` along the sequence dimension
        truncated_labels = labels[:, :input_seq_length]

        non_mask = truncated_labels == -100
        non_mask_indices = non_mask.nonzero(as_tuple=True)

        ones = torch.ones((batch_size, seq_length, vocab_size), device=device)
        zeros = torch.zeros((batch_size, seq_length, vocab_size), device=device)
        
        known_gene_indices = input_ids[non_mask]

        ones[non_mask_indices[0], non_mask_indices[1], :] = 0.0
        zeros[non_mask_indices[0], non_mask_indices[1], known_gene_indices] = 1.0

        # Modify already known genes' probabilities using the one-hot tensor
        modified_probs = probs * ones
        modified_probs = modified_probs + zeros

        # Do the normalization
        modified_probs = modified_probs / modified_probs.sum(dim=-1, keepdim=True).clamp(min=1e-18)  # Normalize

        return modified_probs

    def compute_similarity_on_probs(self, probs: Tensor) -> Tensor:
        """
        Optimized computation of average cosine similarity across all positions in each sequence and batch.

        Args:
            probs (torch.Tensor): Probability tensor of shape (batch_size, seq_length, vocab_size).
            
        Returns:
            torch.Tensor: Average similarity term for loss computation.
        """
        batch_size, seq_length, vocab_size = probs.size()

        # Normalize along the vocab_size dimension
        probs_norm = F.normalize(probs, dim=-1)  # Shape: (batch_size, seq_length, vocab_size)
        
        # Compute pairwise cosine similarity using einsum
        similarities = torch.einsum("biv,bjv->bij", probs_norm, probs_norm)  # Shape: (batch_size, seq_length, seq_length), listing pair-wise similarity values across all positions

        # Mask out lower triangle (to consider only i < j pairs)
        mask_sim = torch.triu(torch.ones(seq_length, seq_length, device=probs.device), diagonal=1)
        valid_similarities = similarities * mask_sim  # Shape: (batch_size, seq_length, seq_length)

        # Compute average similarity
        total_similarity = valid_similarities.sum()
        total_comparisons = mask_sim.sum().item() * batch_size

        return total_similarity / total_comparisons


    def forward(
        self, 
        input_ids: Tensor | None = None, 
        attention_mask: Tensor | None = None, 
        token_type_ids: Tensor | None = None, 
        position_ids: Tensor | None = None, 
        head_mask: Tensor | None = None, 
        inputs_embeds: Tensor | None = None, 
        encoder_hidden_states: Tensor | None = None, 
        encoder_attention_mask: Tensor | None = None, 
        labels: Tensor | None = None, 
        output_attentions: bool | None = None, 
        output_hidden_states: bool | None = None, 
        return_dict: bool | None = None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        
        hidden_states = outputs[0]
        hidden_transform = self.transform(hidden_states)
        logits = self.decoder(hidden_transform) + self.bias

        # temperature = 0.75
        # logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        
        # Probability manipulations to avoid repeats from already known genes
        ### Modified part below
        # print(probs.shape)
        probs = self.assign_known_gene_probs(probs, input_ids, labels)
        convert_probs = self.probability_convert(probs, input_ids, labels)
        assigned_probs = self.assign_known_gene_probs(convert_probs, input_ids, labels)        

        masked_lm_loss = None
        if labels is not None:
            # probs_flat = assigned_probs.view(-1, self.config.vocab_size)  ### Modified
            probs_flat = probs.view(-1, self.config.vocab_size)
            labels_flat = labels.view(-1)
            mask = (labels != -100).float().view(-1)

            # Compute masked cross-entropy loss
            masked_lm_loss = -torch.log(torch.clamp(probs_flat[torch.arange(len(labels_flat)), labels_flat], min=1e-18)) * mask
            masked_lm_loss = masked_lm_loss.sum() / mask.sum()

            similarity_loss = self.compute_similarity_on_probs(assigned_probs)
            lambda_similarity = 200.0  # Adjust this value through experimentation
            masked_lm_loss = masked_lm_loss + lambda_similarity * similarity_loss

            
        else:
            loss = None

        if not return_dict:
            output = (assigned_probs,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            # logits=assigned_probs,
            logits=probs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            )
        
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
