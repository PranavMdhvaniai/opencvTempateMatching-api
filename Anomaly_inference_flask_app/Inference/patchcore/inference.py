import contextlib
import gc
import logging
import os
import sys

import click
import numpy as np
import torch

import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

# Configure the logger
LOGGER = logging.getLogger(__name__)  # Logger with the name of the current module
LOGGER.setLevel(logging.DEBUG)  # Set the log level to DEBUG or any other level

# Create a file handler to write logs to a file
log_file = "inference.log"  # You can choose your own file name
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
LOGGER.addHandler(file_handler)



def get_patchcore_instance(device,patch_core_path, faiss_on_gpu = True, faiss_num_workers = 8):
    n_patchcores = len(
        [x for x in os.listdir(patch_core_path) if ".faiss" in x]
    )
    loaded_patchcores = []
    if n_patchcores == 1:
        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
        patchcore_instance = patchcore.patchcore.PatchCore(device)
        patchcore_instance.load_from_path(
            load_path=patch_core_path, device=device, nn_method=nn_method
        )
        loaded_patchcores.append(patchcore_instance)
    else:
        for i in range(n_patchcores):
            nn_method = patchcore.common.FaissNN(
                faiss_on_gpu, faiss_num_workers
            )
            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load_from_path(
                load_path=patch_core_path,
                device=device,
                nn_method=nn_method,
                prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
            )
            loaded_patchcores.append(patchcore_instance)

    return loaded_patchcores,nn_method