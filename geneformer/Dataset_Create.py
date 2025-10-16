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


# # Randomly select 100_000 sequences from Genecorpus to conduct the training
genecorpus = load_from_disk("/ibex/user/chenj0i/Geneformer/Genecorpus-30M/genecorpus_30M_2048.dataset")

subset_size = 1_200_000
subset_sequences = genecorpus.shuffle(seed=42).select(i for i in tqdm(list(range(subset_size))))['input_ids']
subset_train_dataset = Dataset.from_dict({"input_ids": subset_sequences[:-200_000]})
subset_train_dataset.save_to_disk("/ibex/user/chenj0i/Geneformer/subset_1Mtrain_genecorpus.dataset")
subset_test_dataset = Dataset.from_dict({"input_ids": subset_sequences[-200_000:]})
subset_test_dataset.save_to_disk("/ibex/user/chenj0i/Geneformer/subset_200K_1Mtrain_genecorpus.dataset")

# Create length file for the training
# Define the value to repeat
value_to_repeat = 2048
# Define the total number of elements
total_elements = 1_000_000
# Create the list with repeated values
data_list = [value_to_repeat] * total_elements
# Define the path for the output .pkl length file
output_file = "sub_1Mtrain_genecorpus_30M_2048_lengths.pkl"
# Save the list to a .pkl file
with open(output_file, 'wb') as f:
    pickle.dump(data_list, f)
print(f"List with {subset_size} elements saved as {output_file}")

value_to_repeat_test = 2048
# Define the total number of elements
total_elements_test = 200_000
# Create the list with repeated values
data_list_test = [value_to_repeat_test] * total_elements_test
# Define the path for the output .pkl length file
output_file_test = "sub_200K_1Mtrain_genecorpus_30M_2048_lengths.pkl"
# Save the list to a .pkl file
with open(output_file_test, 'wb') as f:
    pickle.dump(data_list_test, f)
print(f"List with {subset_size} elements saved as {output_file_test}")