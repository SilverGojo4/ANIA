# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-nested-blocks
"""
Deep Learning Training Module for ANIA

This module provides functions for training the ANIA deep learning model in the AMP-MIC project.
It includes the main training pipeline and model training logic with Grid Search, validation,
and early stopping, specifically for FCGR features.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import random
import sys
import time
from itertools import product
from typing import Optional

# ============================== Third-Party Library Imports ==============================
import matplotlib.pyplot as plt
import torch
from matplotlib import style
from torch import nn
from torch.amp import GradScaler, autocast  # type: ignore
from torch.optim import SGD, Adam, AdamW  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# Deep Learning Utility Functions
from models.deep_learning.ANIA import ANIA

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger
from src.models.deep_learning.utils import (
    extract_cgr_features_and_target_for_dl,
    get_hyperparameter_settings,
    read_json_config,
)


# ============================== Custom Function ==============================
def check_cuda_and_optimize(device: str, logger: CustomLogger) -> tuple[list[int], str]:
    """
    Check CUDA availability and set up optimizations for GPU training.

    Parameters
    ----------
    device : str
        Device to run the model on (e.g., 'cuda', 'cuda:0', 'cuda:0,1').
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    tuple[list[int], str]
        A tuple containing:
        - List of GPU IDs to use (e.g., [0, 1]).
        - Primary device (e.g., 'cuda:0').
    """
    # Check CUDA availability
    if "cuda" not in device.lower() or not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required for training, but CUDA is not available. Terminating training."
        )

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    logger.log_with_borders(
        level=logging.INFO,
        message=f"Number of 'available' GPUs: {num_gpus}",
        border="|",
        length=100,
    )

    # Parse the device string
    device = device.lower()
    if device == "cuda":
        # Use all available GPUs
        device_ids = list(range(num_gpus))
        primary_device = "cuda:0"
    elif "," in device:
        # Use specified GPUs (e.g., 'cuda:0,1')
        try:
            device_ids = [int(idx) for idx in device.replace("cuda:", "").split(",")]
            primary_device = f"cuda:{device_ids[0]}"
        except ValueError:
            raise ValueError(
                f"Invalid device format: '{device}'. Expected format: 'cuda:<index1>,<index2>' (e.g., 'cuda:0,1')."
            )
    else:
        # Use a single specified GPU (e.g., 'cuda:0')
        try:
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            device_ids = [device_idx]
            primary_device = device
        except (IndexError, ValueError):
            raise ValueError(
                f"Invalid device format: '{device}'. Expected format: 'cuda:<index>' (e.g., 'cuda:0')."
            )

    # Validate device indices
    for idx in device_ids:
        if idx < 0 or idx >= num_gpus:
            raise ValueError(
                f"Invalid GPU index: {idx}. Available GPUs: 0 to {num_gpus-1}."
            )

    # Set the primary device
    torch.cuda.set_device(device_ids[0])
    gpu_id = torch.cuda.current_device()
    logger.log_with_borders(
        level=logging.INFO,
        message=f"Using GPU devices:\n"
        f"  • Specified device: '{device}'\n"
        f"  • Device IDs: {device_ids}\n"
        f"  • 'Primary GPU' ID: {gpu_id}\n"
        f"  • Total Memory ('Primary GPU'): {torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3):.2f} GB\n"
        f"  • Initial Allocated Memory ('Primary GPU'): {torch.cuda.memory_allocated(gpu_id) / (1024**3):.2f} GB",
        border="|",
        length=100,
    )
    logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    # Enable cuDNN optimization for faster convolution operations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    logger.log_with_borders(
        level=logging.INFO,
        message="'cuDNN' benchmark mode enabled for optimized convolution performance\n"
        f"'cuDNN' enabled: {torch.backends.cudnn.enabled}, 'Benchmark': {torch.backends.cudnn.benchmark}",
        border="|",
        length=100,
    )

    return device_ids, primary_device


def train_ania(
    X_train_fcgr: torch.Tensor,
    y_train: torch.Tensor,
    hyperparams_config_path: str,
    model_output_path: str,
    logger: CustomLogger,
    train_split: float = 0.8,
    patience: int = 10,
    device: str = "cuda:0",
    random_search: bool = False,
    num_random_samples: int = 50,
) -> Optional[nn.Module]:
    """
    Train the ANIA model with Grid Search, validation, and early stopping, and save it.

    Parameters
    ----------
    X_train_fcgr : torch.Tensor
        FCGR training feature set of shape (batch_size, in_channels, height, width).
    y_train : torch.Tensor
        Training target variable of shape (batch_size, 1).
    hyperparams_config_path : str
        Path to the JSON file containing hyperparameter ranges.
    model_output_path : str
        Path to save the trained model.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    train_split : float, optional
        Proportion of data to use for training (default is 0.8).
    patience : int, optional
        Number of epochs to wait for improvement in validation loss before early stopping (default is 10).
    device : str, optional
        Device to run the model on (e.g., 'cuda', 'cuda:0', 'cuda:0,1', default is 'cuda:0').
    random_search : bool, optional
        Whether to perform random search instead of full grid search (default is False).
    num_random_samples : int, optional
        Number of random hyperparameter combinations to sample if random_search is True (default is 50).

    Returns
    -------
    Optional[nn.Module]
        Trained ANIA model instance, or None if training fails.
    """
    try:
        # Log training start
        logger.info(msg=f"/ Task: Train 'ANIA' model")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Check CUDA availability and set up optimizations
        device_ids, primary_device = check_cuda_and_optimize(device, logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record start time for the entire training process
        start_time = time.time()

        # Load hyperparameters
        config = read_json_config(hyperparams_config_path, logger)
        param_grid = get_hyperparameter_settings(config, "ania", logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Split data into training and validation sets
        num_samples = y_train.size(0)
        train_size = int(train_split * num_samples)
        val_size = num_samples - train_size
        X_train_split, X_val = torch.split(X_train_fcgr, [train_size, val_size])
        y_train_split, y_val = torch.split(y_train, [train_size, val_size])

        # Move data to the primary device
        X_train_split = X_train_split.to(primary_device)
        y_train_split = y_train_split.to(primary_device)
        X_val = X_val.to(primary_device)
        y_val = y_val.to(primary_device)
        train_dataset = TensorDataset(X_train_split, y_train_split)
        val_dataset = TensorDataset(X_val, y_val)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Moved training and validation data to '{primary_device}'\n"
            f"Data split:\n"
            f"  • Training samples: {train_size}\n"
            f"  • Validation samples: {val_size}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Calculate total number of hyperparameter combinations
        total_all_combinations = 1
        for values in param_grid.values():
            total_all_combinations *= len(values)
        total_hyperparameters = len(param_grid)

        # Apply Random Search: sample a fixed number of combinations randomly
        if random_search:
            sample_size = num_random_samples
            param_combinations = [
                dict(zip(param_grid.keys(), values))
                for values in [
                    tuple(random.choice(values) for values in param_grid.values())
                    for _ in range(sample_size)
                ]
            ]
            total_combinations_to_show = sample_size
        else:
            param_combinations = (
                dict(zip(param_grid.keys(), values))
                for values in product(*param_grid.values())
            )
            total_combinations_to_show = total_all_combinations
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Performing {'RandomSearch' if random_search else 'GridSearch'} for 'ANIA':\n"
            f"  • Total number of hyperparameters: {total_hyperparameters}\n"
            f"  • Number of hyperparameter combinations: {total_all_combinations}\n"
            f"  • Sampled combinations: {total_combinations_to_show}\n"
            f"  • Patience: {patience}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Initialize variables to track the best model and its performance
        best_val_loss = float("inf")
        best_model = None
        best_params = None
        best_combination_idx = -1
        all_train_losses = []
        all_val_losses = []

        # Initialize GradScaler for mixed precision training with 'cuda' device
        scaler = GradScaler("cuda")

        # Perform grid search over hyperparameter combinations with a progress bar
        logger.log_with_borders(
            level=logging.INFO,
            message="['Training Overview']",
            border="|",
            length=100,
        )
        for idx, param_dict in tqdm(
            enumerate(param_combinations, 1),
            total=float(total_combinations_to_show),
            desc=(
                "Grid Search Progress"
                if not random_search
                else "Random Search Progress"
            ),
        ):
            combination_start_time = time.time()
            train_loader = DataLoader(
                train_dataset,
                batch_size=param_dict["batch_size"],
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=param_dict["batch_size"],
                shuffle=False,
            )

            # Dynamically compute the total output channels for each Inception module
            inception1_out_channels = (
                param_dict["inception1_branch1x1_channels"]
                + param_dict["inception1_branch3x3_channels"]
                + param_dict["inception1_branch5x5_channels"]
                + param_dict["inception1_branch_pool_channels"]
            )
            inception2_out_channels = (
                param_dict["inception2_branch1x1_channels"]
                + param_dict["inception2_branch3x3_channels"]
                + param_dict["inception2_branch5x5_channels"]
                + param_dict["inception2_branch_pool_channels"]
            )

            # Initialize the ANIA model
            model = ANIA(
                in_channels=11,
                inception1_out_channels=inception1_out_channels,
                inception2_out_channels=inception2_out_channels,
                inception1_branch1x1_channels=param_dict[
                    "inception1_branch1x1_channels"
                ],
                inception1_branch3x3_channels=param_dict[
                    "inception1_branch3x3_channels"
                ],
                inception1_branch3x3_reduction=param_dict[
                    "inception1_branch3x3_reduction"
                ],
                inception1_branch5x5_channels=param_dict[
                    "inception1_branch5x5_channels"
                ],
                inception1_branch5x5_reduction=param_dict[
                    "inception1_branch5x5_reduction"
                ],
                inception1_branch_pool_channels=param_dict[
                    "inception1_branch_pool_channels"
                ],
                inception2_branch1x1_channels=param_dict[
                    "inception2_branch1x1_channels"
                ],
                inception2_branch3x3_channels=param_dict[
                    "inception2_branch3x3_channels"
                ],
                inception2_branch3x3_reduction=param_dict[
                    "inception2_branch3x3_reduction"
                ],
                inception2_branch5x5_channels=param_dict[
                    "inception2_branch5x5_channels"
                ],
                inception2_branch5x5_reduction=param_dict[
                    "inception2_branch5x5_reduction"
                ],
                inception2_branch_pool_channels=param_dict[
                    "inception2_branch_pool_channels"
                ],
                num_heads=param_dict["num_heads"],
                d_model=param_dict["d_model"],
                num_encoder_layers=param_dict.get("num_encoder_layers", 2),  # 新增參數
                dense_hidden_dim=param_dict["dense_hidden_dim"],
                dropout_rate=param_dict["dropout_rate"],
            )

            # Move model to the primary device and wrap with DataParallel if multiple GPUs are used
            model = model.to(primary_device)
            if len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)

            # Define the optimizer based on the hyperparameter settings
            optimizer_class = {
                "adam": Adam,
                "adamw": AdamW,
                "sgd": lambda params, lr, weight_decay: SGD(
                    params, lr=lr, momentum=0.9, weight_decay=weight_decay
                ),
            }.get(param_dict["optimizer"], Adam)
            optimizer = optimizer_class(
                model.parameters(),
                lr=param_dict["learning_rate"],
                weight_decay=param_dict["weight_decay"],
            )

            # Define the loss function based on the hyperparameter settings
            criterion_class = {
                "mse": nn.MSELoss,
                "l1": nn.L1Loss,
                "smooth_l1": nn.SmoothL1Loss,
            }.get(param_dict["loss_function"], nn.MSELoss)
            criterion = criterion_class()

            # Initialize learning rate scheduler
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",  # 監控驗證損失，最小化
                factor=0.5,  # 學習率縮減因子（每次降低為原來的 0.5 倍）
                patience=5,  # 等待 5 個 epoch 如果驗證損失未改善
                min_lr=1e-6,  # 學習率最小值
            )

            # Variables for early stopping within this hyperparameter combination
            best_combination_val_loss = float("inf")
            epochs_no_improve = 0
            combination_train_losses = []
            combination_val_losses = []

            # Training loop for the current hyperparameter combination
            for _ in range(param_dict["epochs"]):
                model.train()
                total_train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    with autocast("cuda"):
                        outputs = model(batch_x)
                        train_loss = criterion(outputs, batch_y)
                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_train_loss += train_loss.item()

                # Validation with mixed precision
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    with autocast("cuda"):
                        for batch_x, batch_y in val_loader:
                            val_outputs = model(batch_x)
                            val_loss = criterion(val_outputs, batch_y)
                            total_val_loss += val_loss.item()

                avg_train_loss = total_train_loss / len(train_loader)
                avg_val_loss = total_val_loss / len(val_loader)
                combination_train_losses.append(avg_train_loss)
                combination_val_losses.append(avg_val_loss)

                # Update learning rate based on validation loss
                scheduler.step(avg_val_loss)

                # Check for early stopping based on validation loss improvement
                if avg_val_loss < best_combination_val_loss:
                    best_combination_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            # Log the final training and validation loss for this hyperparameter combination
            final_train_loss = combination_train_losses[-1]
            final_val_loss = combination_val_losses[-1]
            combination_time = time.time() - combination_start_time
            logger.log_with_borders(
                level=logging.INFO,
                message=f"  • {idx}/{total_combinations_to_show} Train Loss = {final_train_loss:.4f}, "
                f"Val Loss = {final_val_loss:.4f}, Training Time = {combination_time:.2f} s",
                border="|",
                length=100,
            )

            # Store losses for this combination
            all_train_losses.append(combination_train_losses)
            all_val_losses.append(combination_val_losses)

            # Evaluate final validation loss for this combination
            if best_combination_val_loss < best_val_loss and not torch.isnan(
                torch.tensor(best_combination_val_loss)
            ):
                best_val_loss = best_combination_val_loss
                best_model = model
                best_params = param_dict
                best_combination_idx = idx - 1

        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Check if a best model was found
        if best_model is None:
            raise ValueError(
                f"No valid model was found during {'RandomSearch' if random_search else 'GridSearch'}. All combinations failed to improve validation loss."
            )

        # Process the best model
        best_train_losses = all_train_losses[best_combination_idx]
        best_val_losses = all_val_losses[best_combination_idx]

        # Log the results of the best model
        param_str = "\n".join([f"    ▸ '{k}': {v}" for k, v in best_params.items()])  # type: ignore
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Best results for 'ANIA':\n"
            f"  • Validation Loss: {best_val_loss:.4f}\n"
            f"  • Hyperparameters:\n{param_str}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record end time and final GPU memory usage
        total_time = time.time() - start_time
        final_allocated_memory = torch.cuda.memory_allocated(device_ids[0]) / (1024**3)
        final_reserved_memory = torch.cuda.memory_reserved(device_ids[0]) / (1024**3)
        final_max_memory = torch.cuda.max_memory_allocated(device_ids[0]) / (1024**3)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Total training completed for 'ANIA' model:\n"
            f"  • Time: {total_time:.2f} s\n"
            f"  • Final GPU Memory Usage ('Primary GPU'):\n"
            f"    ▸ Allocated: {final_allocated_memory:.2f} GB\n"
            f"    ▸ Reserved: {final_reserved_memory:.2f} GB\n"
            f"    ▸ Max Allocated: {final_max_memory:.2f} GB",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Save the model state with explainability data
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        model_to_save = (
            best_model.module if isinstance(best_model, nn.DataParallel) else best_model
        )
        torch.save(
            {
                "state_dict": model_to_save.state_dict(),
                "hyperparams": best_params,
                "input_shape": tuple(X_train_fcgr.shape[1:]),
                "train_stats": {
                    "train_losses": best_train_losses,
                    "val_losses": best_val_losses,
                    "best_epoch": len(best_train_losses),
                },
                "val_metrics": {"best_val_loss": best_val_loss},
                "training_time": total_time,
                "torch_version": torch.__version__,
            },
            model_output_path + ".pt",
        )
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Model state saved to\n'{model_output_path}.pt'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return best_model

    except Exception:
        logger.exception(msg="Unexpected error in 'train_ania()' for 'ania'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_train_ania_pipeline(
    base_path: str,
    strain: str,
    logger: CustomLogger,
    train_split: float = 0.8,
    patience: int = 10,
    device: str = "cuda:0",
    random_search: bool = False,
    num_random_samples: int = 50,
    model_output_path: Optional[str] = None,
) -> None:
    """
    Run the deep learning training pipeline for ANIA model across multiple strains.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    strain : str
        Strain to train on. Must be a single strain name (e.g., "Escherichia coli").
    logger : CustomLogger
        Logger for structured logging throughout the pipeline.
    train_split : float, optional
        Proportion of data to use for training (default is 0.8).
    patience : int, optional
        Number of epochs to wait for improvement in validation loss before early stopping (default is 10).
    device : str, optional
        Device to run the model on (e.g., 'cuda:0', default is 'cuda:0').
    random_search : bool, optional
        Whether to perform random search instead of full grid search (default is False).
    num_random_samples : int, optional
        Number of random hyperparameter combinations to sample if random_search is True (default is 50).
    model_output_path : str, optional
        Path to save the trained model (without the '.pt' extension).

    Returns
    -------
    None
    """
    try:
        # Mapping of full strain names to their corresponding suffixes
        all_strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Validate the strain parameter
        if not isinstance(strain, str):
            raise ValueError(
                f"Invalid strain parameter: '{strain}'. Must be a single strain name (str)."
            )
        if strain not in all_strains:
            raise ValueError(
                f"Invalid strain: '{strain}'. Available strains: {list(all_strains.keys())}"
            )

        # Set the strain to train on
        suffix = all_strains[strain]

        # Define the target column and metadata columns
        target_column = "Log MIC Value"
        metadata_columns = ["ID", "Sequence", "Targets"]

        # Path to hyperparameter configuration
        hyperparams_config_path = os.path.join(
            base_path, "configs/dl_hyperparameters.json"
        )

        # Define feature set for ANIA
        feature_set = {
            "name": "CGR",
            "start_idx": 7,
            "end_idx": 2822,
            "height": 16,
            "width": 16,
            "suffix": "_cgr",
        }

        # Validate feature dimensions
        expected_feature_dim = 11 * feature_set["height"] * feature_set["width"]
        actual_feature_dim = feature_set["end_idx"] - feature_set["start_idx"] + 1
        if actual_feature_dim != expected_feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_feature_dim} features "
                f"(11 x {feature_set['height']} x {feature_set['width']}), but got {actual_feature_dim}."
            )

        # Log the strain being trained
        logger.add_divider(level=logging.INFO, length=80, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Strain - '{strain}' -> Model - 'ANIA' -> Feature - '{feature_set['name']}'",
            border="|",
            length=80,
        )
        logger.add_divider(level=logging.INFO, length=80, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)

        # Extract FCGR features
        train_input_csv_file = os.path.join(
            base_path, f"data/processed/split/{suffix}_train.csv"
        )
        _, X_train_fcgr, y_train = extract_cgr_features_and_target_for_dl(
            file_path=train_input_csv_file,
            metadata_columns=metadata_columns,
            target_column=target_column,
            feature_start_idx=feature_set["start_idx"],
            feature_end_idx=feature_set["end_idx"],
            height=feature_set["height"],
            width=feature_set["width"],
        )

        # Define model output path
        if model_output_path is None:
            model_output_path = os.path.join(
                base_path,
                f"experiments/models/{suffix}/ania",
            )

        # Train the model
        train_ania(
            X_train_fcgr=X_train_fcgr,
            y_train=y_train,
            hyperparams_config_path=hyperparams_config_path,
            model_output_path=model_output_path,
            logger=logger,
            train_split=train_split,
            patience=patience,
            device=device,
            random_search=random_search,
            num_random_samples=num_random_samples,
        )

        # Insert a blank line in the log for readability
        logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_train_ania_pipeline()'.")
        raise
