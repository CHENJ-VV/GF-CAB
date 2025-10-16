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

macro_f1_list = []
acc_list = []

iter_step = 2

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

def prepare_data(
    input_data_file,
    output_directory,
    output_prefix,
    split_id_dict=None,
    test_size=None,
    attr_to_split=None,
    attr_to_balance=None,
    max_trials=100,
    pval_threshold=0.1,
):
    """
    Prepare data for cell state or gene classification.

    **Parameters**

    input_data_file : Path
        | Path to directory containing .dataset input
    output_directory : Path
        | Path to directory where prepared data will be saved
    output_prefix : str
        | Prefix for output file
    split_id_dict : None, dict
        | Dictionary of IDs for train and test splits
        | Three-item dictionary with keys: attr_key, train, test
        | attr_key: key specifying name of column in .dataset that contains the IDs for the data splits
        | train: list of IDs in the attr_key column to include in the train split
        | test: list of IDs in the attr_key column to include in the test split
        | For example: {"attr_key": "individual",
        |               "train": ["patient1", "patient2", "patient3", "patient4"],
        |               "test": ["patient5", "patient6"]}
    test_size : None, float
        | Proportion of data to be saved separately and held out for test set
        | (e.g. 0.2 if intending hold out 20%)
        | If None, will inherit from split_sizes["test"] from Classifier
        | The training set will be further split to train / validation in self.validate
        | Note: only available for CellClassifiers
    attr_to_split : None, str
        | Key for attribute on which to split data while balancing potential confounders
        | e.g. "patient_id" for splitting by patient while balancing other characteristics
        | Note: only available for CellClassifiers
    attr_to_balance : None, list
        | List of attribute keys on which to balance data while splitting on attr_to_split
        | e.g. ["age", "sex"] for balancing these characteristics while splitting by patient
        | Note: only available for CellClassifiers
    max_trials : None, int
        | Maximum number of trials of random splitting to try to achieve balanced other attributes
        | If no split is found without significant (p<0.05) differences in other attributes, will select best
        | Note: only available for CellClassifiers
    pval_threshold : None, float
        | P-value threshold to use for attribute balancing across splits
        | E.g. if set to 0.1, will accept trial if p >= 0.1 for all attributes in attr_to_balance
    """

    if test_size is None:
        test_size = oos_test_size

    # prepare data and labels for classification
    data = load_and_filter(filter_data, nproc, input_data_file)

    if classifier == "cell":
        if "label" in data.features:
            logger.error(
                "Column name 'label' must be reserved for class IDs. Please rename column."
            )
            raise
    elif classifier == "gene":
        if "labels" in data.features:
            logger.error(
                "Column name 'labels' must be reserved for class IDs. Please rename column."
            )
            raise

    if (attr_to_split is not None) and (attr_to_balance is None):
        logger.error(
            "Splitting by attribute while balancing confounders requires both attr_to_split and attr_to_balance to be defined."
        )
        raise

    if not isinstance(attr_to_balance, list):
        attr_to_balance = [attr_to_balance]

    if classifier == "cell":
        # remove cell states representing < rare_threshold of cells
        data = remove_rare(
            data, rare_threshold, cell_state_dict["state_key"], nproc
        )
        # downsample max cells and max per class
        data = downsample_and_shuffle(
            data, max_ncells, None, cell_state_dict
        )
        # rename cell state column to "label"
        data = rename_cols(data, cell_state_dict["state_key"])

    # convert classes to numerical labels and save as id_class_dict
    # of note, will label all genes in gene_class_dict
    # if (cross-)validating, genes will be relabeled in column "labels" for each split
    # at the time of training with Classifier.validate
    data, id_class_dict = label_classes(
        classifier, data, None, nproc
    )

    # save id_class_dict for future reference
    id_class_output_path = (
        Path(output_directory) / f"{output_prefix}_id_class_dict"
    ).with_suffix(".pkl")
    with open(id_class_output_path, "wb") as f:
        pickle.dump(id_class_dict, f)

    if split_id_dict is not None:
        data_dict = dict()
        data_dict["train"] = filter_by_dict(
            data, {split_id_dict["attr_key"]: split_id_dict["train"]}, nproc
        )
        data_dict["test"] = filter_by_dict(
            data, {split_id_dict["attr_key"]: split_id_dict["test"]}, nproc
        )
        train_data_output_path = (
            Path(output_directory) / f"{output_prefix}_labeled_train"
        ).with_suffix(".dataset")
        test_data_output_path = (
            Path(output_directory) / f"{output_prefix}_labeled_test"
        ).with_suffix(".dataset")
        data_dict["train"].save_to_disk(str(train_data_output_path))
        data_dict["test"].save_to_disk(str(test_data_output_path))
    elif (test_size is not None) and (classifier == "cell"):
        if 1 > test_size > 0:
            if attr_to_split is None:
                data_dict = data.train_test_split(
                    test_size=test_size,
                    stratify_by_column=None,
                    seed=42,
                )
                train_data_output_path = (
                    Path(output_directory) / f"{output_prefix}_labeled_train"
                ).with_suffix(".dataset")
                test_data_output_path = (
                    Path(output_directory) / f"{output_prefix}_labeled_test"
                ).with_suffix(".dataset")
                data_dict["train"].save_to_disk(str(train_data_output_path))
                data_dict["test"].save_to_disk(str(test_data_output_path))
            else:
                data_dict, balance_df = cu.balance_attr_splits(
                    data,
                    attr_to_split,
                    attr_to_balance,
                    test_size,
                    max_trials,
                    pval_threshold,
                    cell_state_dict["state_key"],
                    nproc,
                )
                balance_df.to_csv(
                    f"{output_directory}/{output_prefix}_train_test_balance_df.csv"
                )
                train_data_output_path = (
                    Path(output_directory) / f"{output_prefix}_labeled_train"
                ).with_suffix(".dataset")
                test_data_output_path = (
                    Path(output_directory) / f"{output_prefix}_labeled_test"
                ).with_suffix(".dataset")
                data_dict["train"].save_to_disk(str(train_data_output_path))
                data_dict["test"].save_to_disk(str(test_data_output_path))
        else:
            data_output_path = (
                Path(output_directory) / f"{output_prefix}_labeled"
            ).with_suffix(".dataset")
            data.save_to_disk(str(data_output_path))
            print(data_output_path)
    else:
        data_output_path = (
            Path(output_directory) / f"{output_prefix}_labeled"
        ).with_suffix(".dataset")
        data.save_to_disk(str(data_output_path))

def load_and_filter(filter_data, nproc, input_data_file):
    data = load_from_disk(input_data_file)
    if filter_data is not None:
        data = filter_by_dict(data, filter_data, nproc)
    return data
# get number of classes for classifier
def get_num_classes(id_class_dict):
    return len(set(id_class_dict.values()))

def filter_by_dict(data, filter_data, nproc):
    for key, value in filter_data.items():

        def filter_data_by_criteria(example):
            return example[key] in value

        data = data.filter(filter_data_by_criteria, num_proc=nproc)
    if len(data) == 0:
        logger.error("No cells remain after filtering. Check filtering criteria.")
        raise
    return data
def remove_rare(data, rare_threshold, label, nproc):
    if rare_threshold > 0:
        total_cells = len(data)
        label_counter = Counter(data[label])
        nonrare_label_dict = {
            label: [k for k, v in label_counter if (v / total_cells) > rare_threshold]
        }
        data = filter_by_dict(data, nonrare_label_dict, nproc)
    return data
def downsample_and_shuffle(data, max_ncells, max_ncells_per_class, cell_state_dict):
    data = data.shuffle(seed=42)
    num_cells = len(data)
    # if max number of cells is defined, then subsample to this max number
    if max_ncells is not None:
        if num_cells > max_ncells:
            data = data.select([i for i in range(max_ncells)])
    if max_ncells_per_class is not None:
        class_labels = data[cell_state_dict["state_key"]]
        random.seed(42)
        subsample_indices = subsample_by_class(class_labels, max_ncells_per_class)
        data = data.select(subsample_indices)
    return data
def rename_cols(data, state_key):
    data = data.rename_column(state_key, "label")
    return data
def label_classes(classifier, data, gene_class_dict, nproc):
    if classifier == "cell":
        label_set = set(data["label"])
    elif classifier == "gene":
        # remove cells without any of the target genes
        def if_contains_label(example):
            a = pu.flatten_list(gene_class_dict.values())
            b = example["input_ids"]
            return not set(a).isdisjoint(b)

        data = data.filter(if_contains_label, num_proc=nproc)
        label_set = gene_class_dict.keys()

        if len(data) == 0:
            logger.error(
                "No cells remain after filtering for target genes. Check target gene list."
            )
            raise

    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
    id_class_dict = {v: k for k, v in class_id_dict.items()}

    def classes_to_ids(example):
        if classifier == "cell":
            example["label"] = class_id_dict[example["label"]]
        elif classifier == "gene":
            example["labels"] = label_gene_classes(
                example, class_id_dict, gene_class_dict
            )
        return example

    data = data.map(classes_to_ids, num_proc=nproc)
    return data, id_class_dict

def train_classifier(
        model_directory,
        num_classes,
        train_data,
        eval_data,
        output_directory,
        predict=False,
        classifier='cell',
        no_eval=False,
        quantize = False,
        freeze_layers=2,
    ):
        """
        Fine-tune model for cell state or gene classification.

        **Parameters**

        model_directory : Path
            | Path to directory containing model
        num_classes : int
            | Number of classes for classifier
        train_data : Dataset
            | Loaded training .dataset input
            | For cell classifier, labels in column "label".
            | For gene classifier, labels in column "labels".
        eval_data : None, Dataset
            | (Optional) Loaded evaluation .dataset input
            | For cell classifier, labels in column "label".
            | For gene classifier, labels in column "labels".
        output_directory : Path
            | Path to directory where fine-tuned model will be saved
        predict : bool
            | Whether or not to save eval predictions from trainer
        """

        ##### Validate and prepare data #####
        train_data, eval_data = validate_and_clean_cols(
            train_data, eval_data, classifier
        )
        
        if (no_eval is True) and (eval_data is not None):
            logger.warning(
                "no_eval set to True; model will be trained without evaluation."
            )
            eval_data = None

        if (classifier == "gene") and (predict is True):
            logger.warning(
                "Predictions during training not currently available for gene classifiers; setting predict to False."
            )
            predict = False

        # ensure not overwriting previously saved model
        saved_model_test = os.path.join(output_directory, "pytorch_model.bin")
        if os.path.isfile(saved_model_test) is True:
            logger.error("Model already saved to this designated output directory.")
            raise
        # make output directory
        # subprocess.call(f"mkdir {output_directory}", shell=True)
        os.makedirs(output_dir, exist_ok=True)

        ##### Load model and training args #####
        model = load_model(
            "CellClassifier",
            num_classes,
            model_directory,
            "train",
            quantize=quantize,
        )
        #############
        pretrained_model = CustomBertForMaskedLM.from_pretrained(model_directory)
        # Extract the word embeddings from the pretrained model
        pretrained_word_embeddings = pretrained_model.bert.embeddings.word_embeddings.weight.clone()
        model.bert.embeddings.word_embeddings.load_state_dict({"weight": pretrained_word_embeddings})    
        ############  
        def_training_args, def_freeze_layers = get_default_train_args(
            model, classifier, train_data, output_directory
        )

        if training_args is not None:
            def_training_args.update(training_args)
        logging_steps = round(
            len(train_data) / def_training_args["per_device_train_batch_size"] / 10
        )
        def_training_args["logging_steps"] = logging_steps
        def_training_args["output_dir"] = output_directory
        if eval_data is None:
            def_training_args["evaluation_strategy"] = "no"
            def_training_args["load_best_model_at_end"] = False
        training_args_init = TrainingArguments(**def_training_args)

        if freeze_layers is not None:
            def_freeze_layers = freeze_layers

        if def_freeze_layers > 0:
            modules_to_freeze = model.bert.encoder.layer[:def_freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        ##### Fine-tune the model #####
        # define the data collator
        if classifier == "cell":
            data_collator = DataCollatorForCellClassification()
        elif self.classifier == "gene":
            data_collator = DataCollatorForGeneClassification()

        # create the trainer
        trainer = Trainer(
            model=model,
            args=training_args_init,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=compute_metrics,
        )

        # train the classifier
        trainer.train()
        trainer.save_model(output_directory)
        if predict is True:
            # make eval predictions and save predictions and metrics
            predictions = trainer.predict(eval_data)
            prediction_output_path = f"{output_directory}/predictions.pkl"
            with open(prediction_output_path, "wb") as f:
                pickle.dump(predictions, f)
            trainer.save_metrics("eval", predictions.metrics)
        return trainer
    
def validate_and_clean_cols(train_data, eval_data, classifier):
    # validate that data has expected label column and remove others
    if classifier == "cell":
        label_col = "label"
    elif classifier == "gene":
        label_col = "labels"

    cols_to_keep = [label_col] + ["input_ids", "length"]
    if label_col not in train_data.column_names:
        logger.error(f"train_data must contain column {label_col} with class labels.")
        raise
    else:
        train_data = remove_cols(train_data, cols_to_keep)

    if eval_data is not None:
        if label_col not in eval_data.column_names:
            logger.error(
                f"eval_data must contain column {label_col} with class labels."
            )
            raise
        else:
            eval_data = remove_cols(eval_data, cols_to_keep)
    return train_data, eval_data
    
def remove_cols(data, cols_to_keep):
    other_cols = list(data.features.keys())
    other_cols = [ele for ele in other_cols if ele not in cols_to_keep]
    data = data.remove_columns(other_cols)
    return data

def load_model(model_type, num_classes, model_directory, mode, quantize=False):
    if model_type == "MTLCellClassifier-Quantized":
        model_type = "MTLCellClassifier"
        quantize = True

    output_hidden_states = (mode == "eval")

    # Quantization logic
    if quantize:
        if model_type == "MTLCellClassifier":
            quantize_config = BitsAndBytesConfig(load_in_8bit=True)
            peft_config = None
        else:
            quantize_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            peft_config = LoraConfig(
                lora_alpha=128,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="TokenClassification",
            )
    else:
        quantize_config = None
        peft_config = None

    # Model class selection
    model_classes = {
        "Pretrained": BertForMaskedLM,
        "GeneClassifier": BertForTokenClassification,
        "CellClassifier": BertForSequenceClassification,
        "MTLCellClassifier": BertForMaskedLM
    }

    model_class = model_classes.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")

    # Model loading
    model_args = {
        "pretrained_model_name_or_path": model_directory,
        "output_hidden_states": output_hidden_states,
        "output_attentions": False,
    }

    if model_type != "Pretrained":
        model_args["num_labels"] = num_classes

    if quantize_config:
        model_args["quantization_config"] = quantize_config
    
    # Load the model
    model = model_class.from_pretrained(**model_args)
    ###########################

    if mode == "eval":
        model.eval()

    # Handle device placement and PEFT
    if not quantize:
        # Only move non-quantized models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    elif peft_config:
        # Apply PEFT for quantized models (except MTLCellClassifier)
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

    return model

def get_default_train_args(model, classifier, data, output_dir):
    num_layers = quant_layers(model)
    freeze_layers_get = 0
    batch_size = 12
    if classifier == "cell":
        epochs = 10
        evaluation_strategy = "epoch"
        load_best_model_at_end = True
    else:
        epochs = 1
        evaluation_strategy = "no"
        load_best_model_at_end = False

    if num_layers == 6:
        default_training_args = {
            "learning_rate": 5e-5,
            "lr_scheduler_type": "linear",
            "warmup_steps": 500,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
        }
    else:
        default_training_args = {
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
        }

    training_args = {
        "num_train_epochs": epochs,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": evaluation_strategy,
        "logging_steps": np.floor(len(data) / batch_size / 8),  # 8 evals per epoch
        "save_strategy": "epoch",
        "group_by_length": False,
        "length_column_name": "length",
        "disable_tqdm": False,
        "weight_decay": 0.001,
        "load_best_model_at_end": load_best_model_at_end,
    }
    training_args.update(default_training_args)

    return training_args, freeze_layers_get

def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [int(name.split("layer.")[1].split(".")[0])]
    return int(max(layer_nums)) + 1

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
def evaluate_model(
    model,
    num_classes,
    id_class_dict,
    eval_data,
    predict=False,
    output_directory=None,
    output_prefix=None,
):
    """
    Evaluate the fine-tuned model.

    **Parameters**

    model : nn.Module
        | Loaded fine-tuned model (e.g. trainer.model)
    num_classes : int
        | Number of classes for classifier
    id_class_dict : dict
        | Loaded _id_class_dict.pkl previously prepared by Classifier.prepare_data
        | (dictionary of format: numerical IDs: class_labels)
    eval_data : Dataset
        | Loaded evaluation .dataset input
    predict : bool
        | Whether or not to save eval predictions
    output_directory : Path
        | Path to directory where eval data will be saved
    output_prefix : str
        | Prefix for output files
    """

    ##### Evaluate the model #####
    labels = id_class_dict.keys()
    y_pred, y_true, logits_list = classifier_predict(
        model, classifier, eval_data, 100
    )
    conf_mat, macro_f1, acc, roc_metrics = get_metrics(
        y_pred, y_true, logits_list, num_classes, labels
    )
    if predict is True:
        pred_dict = {
            "pred_ids": y_pred,
            "label_ids": y_true,
            "predictions": logits_list,
        }
        pred_dict_output_path = (
            Path(output_directory) / f"{output_prefix}_pred_dict"
        ).with_suffix(".pkl")
        with open(pred_dict_output_path, "wb") as f:
            pickle.dump(pred_dict, f)
    return {
        "conf_mat": conf_mat,
        "macro_f1": macro_f1,
        "acc": acc,
        "roc_metrics": roc_metrics,
    }
        
def classifier_predict(model, classifier_type, evalset, forward_batch_size):
    if classifier_type == "gene":
        label_name = "labels"
    elif classifier_type == "cell":
        label_name = "label"

    predict_logits = []
    predict_labels = []
    model.eval()

    # ensure there is at least 2 examples in each batch to avoid incorrect tensor dims
    evalset_len = len(evalset)
    max_divisible = find_largest_div(evalset_len, forward_batch_size)
    if len(evalset) - max_divisible == 1:
        evalset_len = max_divisible

    max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])

    disable_progress_bar()  # disable progress bar for preprocess_classifier_batch mapping
    for i in trange(0, evalset_len, forward_batch_size):
        max_range = min(i + forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(i, max_range)])
        padded_batch = preprocess_classifier_batch(
            batch_evalset, max_evalset_len, label_name
        )
        padded_batch.set_format(type="torch")

        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]
        label_batch = padded_batch[label_name]
        with torch.no_grad():
            outputs = model(
                input_ids=input_data_batch.to("cuda"),
                attention_mask=attn_msk_batch.to("cuda"),
                labels=label_batch.to("cuda"),
            )
            predict_logits += [torch.squeeze(outputs.logits.to("cpu"))]
            predict_labels += [torch.squeeze(label_batch.to("cpu"))]

    enable_progress_bar()
    logits_by_cell = torch.cat(predict_logits)
    last_dim = len(logits_by_cell.shape) - 1
    all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[last_dim])
    labels_by_cell = torch.cat(predict_labels)
    all_labels = torch.flatten(labels_by_cell)
    logit_label_paired = [
        item
        for item in list(zip(all_logits.tolist(), all_labels.tolist()))
        if item[1] != -100
    ]
    y_pred = [vote(item[0]) for item in logit_label_paired]
    y_true = [item[1] for item in logit_label_paired]
    logits_list = [item[0] for item in logit_label_paired]
    return y_pred, y_true, logits_list

def find_largest_div(N, K):
    rem = N % K
    if rem == 0:
        return N
    else:
        return N - rem
def preprocess_classifier_batch(cell_batch, max_len, label_name):
    if max_len is None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])

    def pad_label_example(example):
        example[label_name] = np.pad(
            example[label_name],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=-100,
        )
        example["input_ids"] = np.pad(
            example["input_ids"],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=gene_token_dict.get("<pad>"),
        )
        example["attention_mask"] = (
            example["input_ids"] != gene_token_dict.get("<pad>")
        ).astype(int)
        return example

    padded_batch = cell_batch.map(pad_label_example)
    return padded_batch
def vote(logit_list):
    m = max(logit_list)
    logit_list.index(m)
    indices = [i for i, x in enumerate(logit_list) if x == m]
    if len(indices) > 1:
        return "tie"
    else:
        return indices[0]
def py_softmax(vector):
    e = np.exp(vector)
    return e / e.sum()
def get_metrics(y_pred, y_true, logits_list, num_classes, labels):
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(labels))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    roc_metrics = None  # roc metrics not reported for multiclass
    if num_classes == 2:
        y_score = [py_softmax(item)[1] for item in logits_list]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_wt = len(tpr)
        roc_auc = auc(fpr, tpr)
        roc_metrics = {
            "fpr": fpr,
            "tpr": tpr,
            "interp_tpr": interp_tpr,
            "auc": roc_auc,
            "tpr_wt": tpr_wt,
        }
    return conf_mat, macro_f1, acc, roc_metrics
def evaluate_saved_model(
    model_directory,
    id_class_dict_file,
    test_data_file,
    output_directory,
    output_prefix,
    predict=True,
):
    """
    Evaluate the fine-tuned model.

    **Parameters**

    model_directory : Path
        | Path to directory containing model
    id_class_dict_file : Path
        | Path to _id_class_dict.pkl previously prepared by Classifier.prepare_data
        | (dictionary of format: numerical IDs: class_labels)
    test_data_file : Path
        | Path to directory containing test .dataset
    output_directory : Path
        | Path to directory where eval data will be saved
    output_prefix : str
        | Prefix for output files
    predict : bool
        | Whether or not to save eval predictions
    """

    # load numerical id to class dictionary (id:class)
    with open(id_class_dict_file, "rb") as f:
        id_class_dict = pickle.load(f)

    # get number of classes for classifier
    num_classes = get_num_classes(id_class_dict)

    # load previously filtered and prepared data
    test_data = load_and_filter(None, nproc, test_data_file)

    # load previously fine-tuned model
    model = load_model(
        "CellClassifier",
        num_classes,
        model_directory,
        "eval",
        quantize=quantize,
    )

    # evaluate the model
    result = evaluate_model(
        model,
        num_classes,
        id_class_dict,
        test_data,
        predict=predict,
        output_directory=output_directory,
        output_prefix="CellClassifier",
    )

    all_conf_mat_df = pd.DataFrame(
        result["conf_mat"],
        columns=id_class_dict.values(),
        index=id_class_dict.values(),
    )
    all_metrics = {
        "conf_matrix": all_conf_mat_df,
        "macro_f1": result["macro_f1"],
        "acc": result["acc"],
    }
    all_roc_metrics = None  # roc metrics not reported for multiclass

    if num_classes == 2:
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = result["roc_metrics"]["interp_tpr"]
        all_roc_auc = result["roc_metrics"]["auc"]
        all_roc_metrics = {
            "mean_tpr": mean_tpr,
            "mean_fpr": mean_fpr,
            "all_roc_auc": all_roc_auc,
        }
    all_metrics["all_roc_metrics"] = all_roc_metrics
    test_metrics_output_path = (
        Path(output_directory) / f"{output_prefix}_test_metrics_dict"
    ).with_suffix(".pkl")
    with open(test_metrics_output_path, "wb") as f:
        pickle.dump(all_metrics, f)

    return all_metrics

def plot_conf_mat(
    conf_mat_dict,
    output_directory,
    output_prefix,
    custom_class_order=None,
):
    """
    Plot confusion matrix results of evaluating the fine-tuned model.

    **Parameters**

    conf_mat_dict : dict
        | Dictionary of model_name : confusion_matrix_DataFrame
        | (all_metrics["conf_matrix"] from self.validate)
    output_directory : Path
        | Path to directory where plots will be saved
    output_prefix : str
        | Prefix for output file
    custom_class_order : None, list
        | List of classes in custom order for plots.
        | Same order will be used for all models.
    """

    for model_name in conf_mat_dict.keys():
        plot_confusion_matrix(
            conf_mat_dict[model_name],
            model_name,
            output_directory,
            output_prefix,
            custom_class_order,
        )
def plot_confusion_matrix(
    conf_mat_df, title, output_dir, output_prefix, custom_class_order
):
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    sns.set(font_scale=1)
    sns.set_style("whitegrid", {"axes.grid": False})
    if custom_class_order is not None:
        conf_mat_df = conf_mat_df.reindex(
            index=custom_class_order, columns=custom_class_order
        )
    display_labels = generate_display_labels(conf_mat_df)
    conf_mat = preprocessing.normalize(conf_mat_df.to_numpy(), norm="l1")
    display = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=display_labels
    )
    display.plot(cmap="Blues", values_format=".2g")
    plt.title(title)
    plt.show()

    output_file = (Path(output_dir) / f"{output_prefix}_conf_mat").with_suffix(".pdf")
    display.figure_.savefig(output_file, bbox_inches="tight")
def generate_display_labels(conf_mat_df):
    display_labels = []
    i = 0
    for label in conf_mat_df.index:
        display_labels += [f"{label}\nn={conf_mat_df.iloc[i,:].sum():.0f}"]
        i = i + 1
    return display_labels

def plot_predictions(
    predictions_file,
    id_class_dict_file,
    title,
    output_directory,
    output_prefix,
    custom_class_order=None,
    kwargs_dict=None,
):
    """
    Plot prediction results of evaluating the fine-tuned model.

    **Parameters**

    predictions_file : path
        | Path of model predictions output to plot
        | (saved output from self.validate if predict_eval=True)
        | (or saved output from self.evaluate_saved_model)
    id_class_dict_file : Path
        | Path to _id_class_dict.pkl previously prepared by Classifier.prepare_data
        | (dictionary of format: numerical IDs: class_labels)
    title : str
        | Title for legend containing class labels.
    output_directory : Path
        | Path to directory where plots will be saved
    output_prefix : str
        | Prefix for output file
    custom_class_order : None, list
        | List of classes in custom order for plots.
        | Same order will be used for all models.
    kwargs_dict : None, dict
        | Dictionary of kwargs to pass to plotting function.
    """
    # load predictions
    with open(predictions_file, "rb") as f:
        predictions = pickle.load(f)

    # load numerical id to class dictionary (id:class)
    with open(id_class_dict_file, "rb") as f:
        id_class_dict = pickle.load(f)

    if isinstance(predictions, dict):
        if all(
            [
                key in predictions.keys()
                for key in ["pred_ids", "label_ids", "predictions"]
            ]
        ):
            # format is output from self.evaluate_saved_model
            predictions_logits = np.array(predictions["predictions"])
            true_ids = predictions["label_ids"]
    else:
        # format is output from self.validate if predict_eval=True
        predictions_logits = predictions.predictions
        true_ids = predictions.label_ids

    num_classes = len(id_class_dict.keys())
    num_predict_classes = predictions_logits.shape[1]
    assert num_classes == num_predict_classes
    classes = id_class_dict.values()
    true_labels = [id_class_dict[idx] for idx in true_ids]
    predictions_df = pd.DataFrame(predictions_logits, columns=classes)
    if custom_class_order is not None:
        predictions_df = predictions_df.reindex(columns=custom_class_order)
    predictions_df["true"] = true_labels
    custom_dict = dict(zip(classes, [i for i in range(len(classes))]))
    if custom_class_order is not None:
        custom_dict = dict(
            zip(custom_class_order, [i for i in range(len(custom_class_order))])
        )
    predictions_df = predictions_df.sort_values(
        by=["true"], key=lambda x: x.map(custom_dict)
    )

    plot_predictions_eu(
        predictions_df, title, output_directory, output_prefix, kwargs_dict
    )
def plot_predictions_eu(predictions_df, title, output_dir, output_prefix, kwargs_dict):
    sns.set(font_scale=2)
    plt.figure(figsize=(10, 10), dpi=150)
    label_colors, label_color_dict = make_colorbar(predictions_df, "true")
    predictions_df = predictions_df.drop(columns=["true"])
    predict_colors_list = [label_color_dict[label] for label in predictions_df.columns]
    predict_label_list = [label for label in predictions_df.columns]
    predict_colors = pd.DataFrame(
        pd.Series(predict_colors_list, index=predict_label_list), columns=["predicted"]
    )

    default_kwargs_dict = {
        "row_cluster": False,
        "col_cluster": False,
        "row_colors": label_colors,
        "col_colors": predict_colors,
        "linewidths": 0,
        "xticklabels": False,
        "yticklabels": False,
        "center": 0,
        "cmap": "vlag",
    }

    if kwargs_dict is not None:
        default_kwargs_dict.update(kwargs_dict)
    g = sns.clustermap(predictions_df, **default_kwargs_dict)

    plt.setp(g.ax_row_colors.get_xmajorticklabels(), rotation=45, ha="right")

    for label_color in list(label_color_dict.keys()):
        g.ax_col_dendrogram.bar(
            0, 0, color=label_color_dict[label_color], label=label_color, linewidth=0
        )

        g.ax_col_dendrogram.legend(
            title=f"{title}",
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, 1),
            facecolor="white",
        )

    output_file = (Path(output_dir) / f"{output_prefix}_pred").with_suffix(".pdf")
    plt.savefig(output_file, bbox_inches="tight")
def make_colorbar(embs_df, label):
    labels = list(embs_df[label])

    cell_type_colors = gen_heatmap_class_colors(labels, embs_df)
    label_colors = pd.DataFrame(cell_type_colors, columns=[label])

    # create dictionary for colors and classes
    label_color_dict = gen_heatmap_class_dict(labels, label_colors[label])
    return label_colors, label_color_dict
def gen_heatmap_class_colors(labels, df):
    pal = sns.cubehelix_palette(
        len(Counter(labels).keys()),
        light=0.9,
        dark=0.1,
        hue=1,
        reverse=True,
        start=1,
        rot=-2,
    )
    lut = dict(zip(map(str, Counter(labels).keys()), pal))
    colors = pd.Series(labels, index=df.index).map(lut)
    return colors
def gen_heatmap_class_dict(classes, label_colors_series):
    class_color_dict_df = pd.DataFrame(
        {"classes": classes, "color": label_colors_series}
    )
    class_color_dict_df = class_color_dict_df.drop_duplicates(subset=["classes"])
    return dict(zip(class_color_dict_df["classes"], class_color_dict_df["color"]))


for i in range(iter_step):

    model_directory = "model path"

    corpus_dir = "Pretrain_data"
    with open(corpus_dir + "/token_dictionary.pkl", "rb") as fp:
        gene_token_dict = pickle.load(fp)
    token_gene_dict = {v: k for k, v in gene_token_dict.items()}

    filter_data_dict={"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]}
    training_args = {
        "num_train_epochs": 0.9,
        "learning_rate": 0.000804,
        "lr_scheduler_type": "polynomial",
        "warmup_steps": 1812,
        "weight_decay":0.258828,
        "per_device_train_batch_size": 12,
        "seed": 73,
    }

    cell_state_dict = {"state_key": "disease", "states": "all"}
    classifier='cell'
    filter_data=filter_data_dict
    split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1}
    train_size = split_sizes["train"]
    valid_size = split_sizes["valid"]
    oos_test_size = split_sizes["test"]
    max_ncells=None
    freeze_layers = 2
    num_crossval_splits = 1
    forward_batch_size=200
    nproc=16
    rare_threshold=0
    quantize=None


    train_ids = ["1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678", "1561", "1304", "1610", "1430", "1472", "1707", "1726", "1504", "1425", "1617", "1631", "1735", "1582", "1722", "1622", "1630", "1290", "1479", "1371", "1549", "1515"]
    eval_ids = ["1422", "1510", "1539", "1606", "1702"]
    test_ids = ["1437", "1516", "1602", "1685", "1718"]

    train_test_id_split_dict = {"attr_key": "individual",
                                "train": train_ids+eval_ids,
                                "test": test_ids}
    train_valid_id_split_dict = {"attr_key": "individual",
                                "train": train_ids,
                                "eval": eval_ids}

    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.strftime('%X').replace(':','')}"
    datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_directory = "output path"

    if output_directory[-1:] != "/":  # add slash for dir if not present
        output_directory = output_directory + "/"
    output_dir = f"{output_directory}{datestamp}_geneformer_diseaseClassifier/"
    output_prefix = "cm_classifier_test"
    subprocess.call(f"mkdir {output_dir}", shell=True)
    os.makedirs(output_dir, exist_ok=True)

    prepare_data(input_data_file="example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset",
                    output_directory=output_dir,
                    output_prefix=output_prefix,
                    split_id_dict=train_test_id_split_dict)

    with open(f"{output_dir}/{output_prefix}_id_class_dict.pkl", "rb") as f:
        id_class_dict = pickle.load(f)
    class_id_dict = {v: k for k, v in id_class_dict.items()}

    num_classes = get_num_classes(id_class_dict)

    data = load_and_filter(None, nproc, f"{output_dir}/{output_prefix}_labeled_train.dataset")
    data = data.shuffle(seed=42)

    ##### (Cross-)validate the model #####
    results = []
    all_conf_mat = np.zeros((num_classes, num_classes))
    iteration_num = 1
    split_id_dict=train_valid_id_split_dict

    for i in trange(num_crossval_splits):
        print(
            f"****** Validation split: {iteration_num}/{num_crossval_splits} ******\n"
        )
        ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
        if num_crossval_splits == 1:
            # single 1-eval_size:eval_size split
            if split_id_dict is not None:
                data_dict = dict()
                data_dict["train"] = filter_by_dict(
                    data,
                    {split_id_dict["attr_key"]: split_id_dict["train"]},
                    nproc,
                )
                data_dict["test"] = filter_by_dict(
                    data,
                    {split_id_dict["attr_key"]: split_id_dict["eval"]},
                    nproc,
                )
            train_data = data_dict["train"]
            eval_data = data_dict["test"]

    trainer = train_classifier(
        model_directory,
        num_classes,
        train_data,
        eval_data,
        ksplit_output_dir,
    )

    result = evaluate_model(
                        trainer.model,
                        num_classes,
                        id_class_dict,
                        eval_data,
                        True,
                        ksplit_output_dir,
                        output_prefix,
                    )
    results += [result]
    all_conf_mat = all_conf_mat + result["conf_mat"]
    iteration_num = iteration_num + 1

    all_conf_mat_df = pd.DataFrame(
        all_conf_mat, columns=id_class_dict.values(), index=id_class_dict.values()
    )
    all_metrics = {
        "conf_matrix": all_conf_mat_df,
        "macro_f1": [result["macro_f1"] for result in results],
        "acc": [result["acc"] for result in results],
    }
    all_roc_metrics = None  # roc metrics not reported for multiclass
    if num_classes == 2:
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = [result["roc_metrics"]["interp_tpr"] for result in results]
        all_roc_auc = [result["roc_metrics"]["auc"] for result in results]
        all_tpr_wt = [result["roc_metrics"]["tpr_wt"] for result in results]
        mean_tpr, roc_auc, roc_auc_sd = eu.get_cross_valid_roc_metrics(
            all_tpr, all_roc_auc, all_tpr_wt
        )
        all_roc_metrics = {
            "mean_tpr": mean_tpr,
            "mean_fpr": mean_fpr,
            "all_roc_auc": all_roc_auc,
            "roc_auc": roc_auc,
            "roc_auc_sd": roc_auc_sd,
        }
    all_metrics["all_roc_metrics"] = all_roc_metrics
    save_eval_output=True
    if save_eval_output is True:
        eval_metrics_output_path = (
            Path(output_dir) / f"cm_classifier_test_eval_metrics_dict"
        ).with_suffix(".pkl")
        with open(eval_metrics_output_path, "wb") as f:
            pickle.dump(all_metrics, f)

    datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
    all_metrics_test = evaluate_saved_model(
            model_directory=f"{output_dir}/ksplit1/",
            id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
            test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
            output_directory=output_dir,
            output_prefix=output_prefix,
        )
    
    macro_f1_list.append(all_metrics_test['macro_f1'])
    acc_list.append(all_metrics_test['acc'])


print("Macro F1: ", macro_f1_list)
print("Accuracy: ", acc_list)
