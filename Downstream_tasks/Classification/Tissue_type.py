import os
from tqdm.auto import tqdm, trange
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# imports
from collections import Counter
import seaborn as sns; sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer
from transformers.training_args import TrainingArguments
import pandas as pd
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from sklearn import preprocessing
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from pathlib import Path

import sys
# sys.path.append('../Geneformer')
from geneformer import DataCollatorForCellClassification
from datasets import load_from_disk
import sys
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
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
from transformers import (BertConfig, BertForMaskedLM, TrainingArguments, TrainerCallback, 
                        Trainer, BertModel, BertPreTrainedModel, BertForSequenceClassification, BertForTokenClassification)
from geneformer import GeneformerPretrainer
from torch import Tensor
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertOnlyMLMHead, BertPredictionHeadTransform
from transformers.activations import ACT2FN
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F

model_path = 'model path'
prefix = 'CAB5_1M'
total_iter = 1

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

        # temperature = 0.75
        # logits = logits / temperature

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
            lambda_similarity = 5.0  # Adjust this value through experimentation
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


# load cell type dataset (includes all tissues)
train_dataset=load_from_disk("example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset")
# load evaluation dataset (includes all tissues)
eval_dataset=load_from_disk("example_input_files/cell_classification/cell_type_annotation/cell_type_test_data.dataset")

dataset_list = []
evalset_list = []
organ_list = []
target_dict_list = []

for organ in Counter(train_dataset["organ_major"]).keys():
    # collect list of tissues for fine-tuning (immune and bone marrow are included together)
    if organ in ["bone_marrow"]:  
        continue
    elif organ=="immune":
        organ_ids = ["immune","bone_marrow"]
        organ_list += ["immune"]
    else:
        organ_ids = [organ]
        organ_list += [organ]
    
    # filter datasets for given organ
    def if_organ(example):
        return example["organ_major"] in organ_ids
    trainset_organ = train_dataset.filter(if_organ, num_proc=16)
    
    # per scDeepsort published method, drop cell types representing <0.5% of cells
    celltype_counter = Counter(trainset_organ["cell_type"])
    total_cells = sum(celltype_counter.values())
    cells_to_keep = [k for k,v in celltype_counter.items() if v>(0.005*total_cells)]
    def if_not_rare_celltype(example):
        return example["cell_type"] in cells_to_keep
    trainset_organ_subset = trainset_organ.filter(if_not_rare_celltype, num_proc=16)
    
    # shuffle datasets and rename columns
    trainset_organ_shuffled = trainset_organ_subset.shuffle(seed=42)
    trainset_organ_shuffled = trainset_organ_shuffled.rename_column("cell_type","label")
    trainset_organ_shuffled = trainset_organ_shuffled.remove_columns("organ_major")
    
    # create dictionary of cell types : label ids
    target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    target_dict_list += [target_name_id_dict]
    
    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)
    
    # create 80/20 train/eval splits
    labeled_train_split = labeled_trainset.select([i for i in range(0,round(len(labeled_trainset)*0.8))])
    labeled_eval_split = labeled_trainset.select([i for i in range(round(len(labeled_trainset)*0.8),len(labeled_trainset))])
    
    # filter dataset for cell types in corresponding training set
    trained_labels = list(Counter(labeled_train_split["label"]).keys())
    def if_trained_label(example):
        return example["label"] in trained_labels
    labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)

    dataset_list += [labeled_train_split]
    evalset_list += [labeled_eval_split_subset]

trainset_dict = dict(zip(organ_list,dataset_list))
traintargetdict_dict = dict(zip(organ_list,target_dict_list))

evalset_dict = dict(zip(organ_list,evalset_list))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    return {
    'accuracy': acc,
    'macro_f1': macro_f1,
    'weighted_f1': weighted_f1
    }

# set model parameters
# max input size
max_input_size = 2 ** 11  # 2048

# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and eval
geneformer_batch_size = 12
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 10
# optimizer
optimizer = "adamw"

for organ in organ_list:
    print(organ)
    organ_trainset = trainset_dict[organ]
    organ_evalset = evalset_dict[organ]
    organ_label_dict = traintargetdict_dict[organ]
    
    # set logging steps
    logging_steps = round(len(organ_trainset)/geneformer_batch_size/10)
    
    # reload pretrained model
    model = BertForSequenceClassification.from_pretrained(model_path, 
                                                    num_labels=len(organ_label_dict.keys()),
                                                    output_attentions = False,
                                                    output_hidden_states = False).to("cuda")
    
    # #############
    pretrained_model = CustomBertForMaskedLM.from_pretrained(model_path)
    # Extract the word embeddings from the pretrained model
    pretrained_word_embeddings = pretrained_model.bert.embeddings.word_embeddings.weight.clone()
    model.bert.embeddings.word_embeddings.load_state_dict({"weight": pretrained_word_embeddings})    
    # ############  

    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_dir = f"/ibex/user/chenj0i/Geneformer/Downstream_tasks/Cell_Classify/{prefix}/{datestamp}_geneformer_CellClassifier_{organ}/"
    
    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    # subprocess.call(f'mkdir {output_dir}', shell=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
    }
    
    training_args_init = TrainingArguments(**training_args)

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=organ_trainset,
        eval_dataset=organ_evalset,
        compute_metrics=compute_metrics
    )
    # train the cell type classifier
    trainer.train()
    predictions = trainer.predict(organ_evalset)
    with open(f"{output_dir}predictions.pickle", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval",predictions.metrics)
    trainer.save_model(output_dir)