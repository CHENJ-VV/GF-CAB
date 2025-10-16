import numpy as np
from datasets import load_from_disk
import torch
from transformers import BertForMaskedLM
import os
import sys
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from geneformer.pretrainer import token_dictionary

import datetime

# imports
import os
import time

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

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

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# set local time/directories
timezone = pytz.timezone("Asia/Riyadh")
rootdir = os.getcwd() + "/Self_train"


corpus_dir = "Pretrain_data"
with open(corpus_dir + "/token_dictionary.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

len_vocabulary = len(token_dictionary)

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

        self.post_init()

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

    def compute_similarity_on_probs(self, probs: Tensor, labels: Tensor) -> Tensor:
        """
        Optimized computation of average cosine similarity across all positions in each sequence and batch.

        Args:
            probs (torch.Tensor): Probability tensor of shape (batch_size, seq_length, vocab_size).
            
        Returns:
            torch.Tensor: Average similarity term for loss computation.
        """
        batch_size, seq_length, vocab_size = probs.size()
        device = probs.device

        non_mask = labels == -100
        non_mask_indices = non_mask.nonzero(as_tuple=True)

        mask_sim = torch.ones((batch_size, seq_length, seq_length), device=device)
        mask_sim[non_mask_indices[0], non_mask_indices[1], :] = 0.0

        seq_mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        batch_mask = seq_mask.unsqueeze(0).expand(batch_size, seq_length, seq_length)
        mask_sim = mask_sim * batch_mask

        # Normalize along the vocab_size dimension
        probs_norm = F.normalize(probs, dim=-1)  # Shape: (batch_size, seq_length, vocab_size)
        
        # Compute pairwise cosine similarity using einsum
        similarities = torch.einsum("biv,bjv->bij", probs_norm, probs_norm)  # Shape: (batch_size, seq_length, seq_length), listing pair-wise similarity values across all positions

        # Mask out lower triangle (to consider only i < j pairs)
        # mask_sim = torch.triu(torch.ones(seq_length, seq_length, device=probs.device), diagonal=1)
        valid_similarities = similarities * mask_sim  # Shape: (batch_size, seq_length, seq_length)

        # Compute average similarity
        total_similarity = valid_similarities.sum()
        total_comparisons = mask_sim.sum().item()

        if total_comparisons == 0:
            return torch.tensor(0.0, device=device)
        
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

        probs = F.softmax(logits, dim=-1)
        
        # Probability manipulations to avoid repeats from already known genes
        probs = self.assign_known_gene_probs(probs, input_ids, labels)
        convert_probs = self.probability_convert(probs, input_ids, labels)
        assigned_probs = self.assign_known_gene_probs(convert_probs, input_ids, labels)        

        masked_lm_loss = None
        if labels is not None:
            probs_flat = assigned_probs.view(-1, self.config.vocab_size)
            labels_flat = labels.view(-1)
            mask = (labels != -100).float().view(-1)

            # Compute masked cross-entropy loss
            masked_lm_loss = -torch.log(torch.clamp(probs_flat[torch.arange(len(labels_flat)), labels_flat], min=1e-18)) * mask
            masked_lm_loss = masked_lm_loss.sum() / mask.sum()

            similarity_loss = self.compute_similarity_on_probs(assigned_probs, labels)
            lambda_similarity = 1.0  # Adjust this value through experimentation
            masked_lm_loss = masked_lm_loss + lambda_similarity * similarity_loss

            
        else:
            loss = None

        if not return_dict:
            output = (assigned_probs,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=assigned_probs,
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

# set model parameters
# model type
model_type = "bert"
# max input size
max_input_size = 2**11  # 2048
# number of layers
num_layers = 6
# number of attention heads
num_attn_heads = 4
# number of embedding dimensions
num_embed_dim = 256
# intermediate size
intermed_size = num_embed_dim * 2
# activation function
activ_fn = "relu"
# initializer range, layer norm, dropout
initializer_range = 0.02
layer_norm_eps = 1e-12
attention_probs_dropout_prob = 0.02
hidden_dropout_prob = 0.02

# set training parameters
# total number of examples in Genecorpus-30M after QC filtering:
num_examples = 27_406_208
# number gpus
num_gpus = 8
# batch size for training and eval
geneformer_batch_size = 8
# max learning rate
max_lr = 1e-3
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 10_000
# number of epochs
epochs = 3
# optimizer
optimizer = "adamw"
# weight_decay
weight_decay = 0.001


# output directories
current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = f"GF_CAB_{datestamp}_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")


model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")


# make training and model output directories
os.makedirs(training_output_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

# model configuration
config = {
    "hidden_size": num_embed_dim,
    "num_hidden_layers": num_layers,
    "initializer_range": initializer_range,
    "layer_norm_eps": layer_norm_eps,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "hidden_dropout_prob": hidden_dropout_prob,
    "intermediate_size": intermed_size,
    "hidden_act": activ_fn,
    "max_position_embeddings": max_input_size,
    "model_type": model_type,
    "num_attention_heads": num_attn_heads,
    "pad_token_id": token_dictionary.get("<pad>"),
    "vocab_size": len(token_dictionary),  # genes+2 for <mask> and <pad> tokens
}

config = BertConfig(**config)
model = CustomBertForMaskedLM(config)
model = model.train()


training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": False,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay,
    "per_device_train_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "save_strategy": "steps",
    "save_steps": np.floor(num_examples / geneformer_batch_size / 8),  # 8 saves per epoch
    "logging_steps": 1000,
    "output_dir": training_output_dir,
    "logging_dir": logging_dir,
}
training_args = TrainingArguments(**training_args)

print("Starting training.")

# define the trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    train_dataset=load_from_disk("Pretrain_data/genecorpus_30M_2048.dataset"),
    example_lengths_file="Pretrain_data/genecorpus_30M_2048_lengths.pkl",
    token_dictionary=token_dictionary,
)

# train
trainer.train()
# save model
trainer.save_model(model_output_dir)