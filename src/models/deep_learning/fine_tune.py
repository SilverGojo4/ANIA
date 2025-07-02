# pylint: disable=too-many-arguments, line-too-long, import-error, invalid-name, too-many-locals, too-many-branches, wrong-import-position, too-many-positional-arguments, too-many-statements, too-many-nested-blocks
"""
Fine-tuning Module for ANIA Deep Learning Model

This module provides a pipeline for fine-tuning a pretrained ANIA model in the AMP-MIC project.
It loads a previously trained model, freezes selected layers, and performs additional training on the remaining parameters.
Supports only ANIA (2D CGR features).
"""
# ============================== Standard Library Imports ==============================
import argparse
import logging
import os
import sys
import time
from typing import Optional

# ============================== Third-Party Library Imports ==============================
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.amp import GradScaler, autocast  # type: ignore
from torch.optim import SGD, Adam, AdamW  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger, setup_logging

# Deep Learning Utility Functions
from src.models.deep_learning.ANIA import ANIA
from src.models.deep_learning.train import check_cuda_and_optimize
from src.models.deep_learning.utils import extract_cgr_features_and_target_for_dl


def fine_tune_model(
    base_model_path: str,
    X_train_fcgr: torch.Tensor,
    y_train: torch.Tensor,
    output_path: str,
    logger: CustomLogger,
    device: str = "cuda:0",
    epochs: int = 20,
    freeze_part: str = "inception",
    batch_size: int = 128,
    patience: int = 3,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.001,
    dropout_rate: float = 0.5,
    loss_threshold: float = 1e-4,  # 新增：訓練損失變化閾值
) -> Optional[nn.Module]:
    """
    Fine-tune a pretrained ANIA model with specified parameters using the entire dataset for training.

    Parameters
    ----------
    base_model_path : str
        Path to the pretrained model checkpoint.
    X_train_fcgr : torch.Tensor
        FCGR training feature set of shape (batch_size, in_channels, height, width) for 'ANIA'.
    y_train : torch.Tensor
        Training target tensor of shape (batch_size, 1).
    output_path : str
        Path to save the fine-tuned model.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    device : str, optional
        Device to run training on (default: 'cuda:0').
    epochs : int, optional
        Number of fine-tuning epochs (default: 20).
    freeze_part : str, optional
        Part of the model to freeze. Must be one of: "inception", "transformer", "dense", or "none".
        - "inception": Freeze both inception1 and inception2 modules.
        - "transformer": Freeze the transformer_encoder module.
        - "dense": Freeze the dense layer.
        - "none": Do not freeze any layers.
        (default: "inception")
    batch_size : int, optional
        Batch size for training (default: 128).
    patience : int, optional
        Number of epochs to wait for improvement in training loss before early stopping (default: 3).
    learning_rate : float, optional
        Learning rate for fine-tuning (default: 1e-5).
    weight_decay : float, optional
        Weight decay for regularization (default: 0.001).
    dropout_rate : float, optional
        Dropout rate for regularization (default: 0.5).
    loss_threshold : float, optional
        Threshold for training loss change to trigger early stopping (default: 1e-4).

    Returns
    -------
    Optional[nn.Module]
        Fine-tuned ANIA model instance, or None if fine-tuning fails.
    """
    try:
        # Log fine-tuning start
        logger.info(msg="/ Task: Starting 'Fine-Tuning' phase for 'ANIA' model")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Check device and set up optimizations
        device_ids, primary_device = check_cuda_and_optimize(device, logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Move data to device and ensure correct dtype
        X_train_fcgr = X_train_fcgr.to(primary_device, dtype=torch.float32)
        y_train = y_train.to(primary_device, dtype=torch.float32)

        # Load pretrained model checkpoint
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading pretrained 'ANIA' model from checkpoint:\n'{base_model_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(
                f"Pretrained model checkpoint not found at '{base_model_path}'"
            )
        checkpoint = torch.load(base_model_path, map_location=primary_device)
        best_hparams = checkpoint["hyperparams"]
        input_shape = checkpoint["input_shape"]
        loss_name = best_hparams.get("loss_function", "mse")

        # Use the entire dataset for training (no validation split)
        train_dataset = TensorDataset(X_train_fcgr, y_train)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Using entire dataset for fine-tuning:\n"
            f"  • Total samples: {len(train_dataset)}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Dynamically compute the total output channels for each Inception module
        inception1_out_channels = (
            best_hparams["inception1_branch1x1_channels"]
            + best_hparams["inception1_branch3x3_channels"]
            + best_hparams["inception1_branch5x5_channels"]
            + best_hparams["inception1_branch_pool_channels"]
        )
        inception2_out_channels = (
            best_hparams["inception2_branch1x1_channels"]
            + best_hparams["inception2_branch3x3_channels"]
            + best_hparams["inception2_branch5x5_channels"]
            + best_hparams["inception2_branch_pool_channels"]
        )

        # Rebuild ANIA model
        model = ANIA(
            in_channels=11,
            inception1_out_channels=inception1_out_channels,
            inception2_out_channels=inception2_out_channels,
            inception1_branch1x1_channels=best_hparams["inception1_branch1x1_channels"],
            inception1_branch3x3_channels=best_hparams["inception1_branch3x3_channels"],
            inception1_branch3x3_reduction=best_hparams[
                "inception1_branch3x3_reduction"
            ],
            inception1_branch5x5_channels=best_hparams["inception1_branch5x5_channels"],
            inception1_branch5x5_reduction=best_hparams[
                "inception1_branch5x5_reduction"
            ],
            inception1_branch_pool_channels=best_hparams[
                "inception1_branch_pool_channels"
            ],
            inception2_branch1x1_channels=best_hparams["inception2_branch1x1_channels"],
            inception2_branch3x3_channels=best_hparams["inception2_branch3x3_channels"],
            inception2_branch3x3_reduction=best_hparams[
                "inception2_branch3x3_reduction"
            ],
            inception2_branch5x5_channels=best_hparams["inception2_branch5x5_channels"],
            inception2_branch5x5_reduction=best_hparams[
                "inception2_branch5x5_reduction"
            ],
            inception2_branch_pool_channels=best_hparams[
                "inception2_branch_pool_channels"
            ],
            num_heads=best_hparams["num_heads"],
            d_model=best_hparams["d_model"],
            num_encoder_layers=best_hparams.get("num_encoder_layers", 1),
            dense_hidden_dim=best_hparams["dense_hidden_dim"],
            dropout_rate=dropout_rate,
        ).to(primary_device)

        # Load pretrained weights
        model.load_state_dict(checkpoint["state_dict"])
        logger.log_with_borders(
            level=logging.INFO,
            message="Successfully loaded pretrained weights.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Validate freeze_part and freeze the specified part
        valid_parts = ["inception", "transformer", "dense", "none"]
        if freeze_part not in valid_parts:
            raise ValueError(
                f"Invalid freeze_part '{freeze_part}'. Valid options are: {valid_parts}"
            )

        frozen_layers = []
        if freeze_part == "inception":
            for param in model.inception1.parameters():
                param.requires_grad = False
            for param in model.inception2.parameters():
                param.requires_grad = False
            frozen_layers.extend(["inception1", "inception2"])
        elif freeze_part == "transformer":
            for param in model.transformer_encoder.parameters():
                param.requires_grad = False
            frozen_layers.append("transformer")
        elif freeze_part == "dense":
            for param in model.dense.parameters():
                param.requires_grad = False
            frozen_layers.append("dense")
        # If freeze_part is "none", no layers are frozen

        if frozen_layers:
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Froze parameters of the following layers: {frozen_layers}",
                border="|",
                length=100,
            )
        else:
            logger.log_with_borders(
                level=logging.INFO,
                message="No layers were frozen.",
                border="|",
                length=100,
            )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Setup optimizer
        optimizer_cls = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": lambda params, lr, weight_decay: SGD(
                params, lr=lr, weight_decay=weight_decay, momentum=0.9
            ),
        }.get(best_hparams["optimizer"], AdamW)
        optimizer = optimizer_cls(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )

        # Setup loss function
        loss_fn = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "smooth_l1": nn.SmoothL1Loss(),
        }.get(loss_name, nn.MSELoss())

        # Setup DataLoader for training (using the entire dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        scaler = GradScaler("cuda")
        train_losses = []
        start_time = time.time()

        # Early stopping variables (based on training loss)
        best_train_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        # Fine-tuning loop with early stopping based on training loss
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                with autocast("cuda"):
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_train_loss += loss.item()
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Adjust learning rate based on training loss
            scheduler.step(avg_train_loss)

            # Log current fine-tuning metrics
            current_lr = optimizer.param_groups[0]["lr"]
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, "
                f"Learning Rate = {current_lr:.6f}",
                border="|",
                length=100,
            )

            # Early stopping check based on training loss improvement
            if avg_train_loss < best_train_loss:
                improvement = best_train_loss - avg_train_loss
                best_train_loss = avg_train_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                if improvement < loss_threshold:  # 如果損失改善小於閾值，視為收斂
                    logger.log_with_borders(
                        level=logging.INFO,
                        message=f"Training loss improvement ({improvement:.6f}) below threshold ({loss_threshold}). Stopping training.",
                        border="|",
                        length=100,
                    )
                    break
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.log_with_borders(
                        level=logging.INFO,
                        message=f"Early stopping triggered after {epoch + 1} epochs (no improvement in training loss).",
                        border="|",
                        length=100,
                    )
                    break

        # Restore the best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Plot training loss curve (no validation loss)
        loss_plot_path = output_path + "_loss_curve.png"
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label="Train Loss",
            color="#82B0D2",
            linewidth=2,
        )
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record end time and final GPU memory usage
        total_time = time.time() - start_time
        if "cuda" in primary_device and torch.cuda.is_available():
            final_allocated_memory = torch.cuda.memory_allocated(device_ids[0]) / (
                1024**3
            )
            final_reserved_memory = torch.cuda.memory_reserved(device_ids[0]) / (
                1024**3
            )
            final_max_memory = torch.cuda.max_memory_allocated(device_ids[0]) / (
                1024**3
            )
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Fine-tuning completed for 'ANIA' model:\n"
                f"  • Time: {total_time:.2f}s\n"
                f"  • Final GPU Memory Usage:\n"
                f"    ▸ Allocated: {final_allocated_memory:.2f} GB\n"
                f"    ▸ Reserved: {final_reserved_memory:.2f} GB\n"
                f"    ▸ Max Allocated: {final_max_memory:.2f} GB",
                border="|",
                length=100,
            )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Save fine-tuned model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_dict = {
            "state_dict": model.state_dict(),
            "hyperparams": best_hparams,
            "input_shape": input_shape,
            "fine_tune_stats": {
                "train_losses": train_losses,
                "epochs": len(train_losses),
                "fine_tune_time": total_time,
            },
            "torch_version": torch.__version__,
        }
        torch.save(save_dict, output_path + ".pt")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved fine-tuned model to\n'{output_path}.pt'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return model

    except Exception:
        logger.exception(msg="Unexpected error in 'fine_tune_model()' for 'ANIA'.")
        raise


def run_fine_tune_pipeline(
    base_path: str,
    strain: str,
    logger: CustomLogger,
    train_input_file: str,
    ft_epochs: int = 20,
    device: str = "cuda:0",
    freeze_part: str = "inception",
    patience: int = 3,
    base_model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.001,
    dropout_rate: float = 0.5,
    batch_size: int = 128,
    loss_threshold: float = 1e-4,  # 新增：訓練損失變化閾值
) -> None:
    """
    Run the fine-tuning pipeline for a pretrained ANIA model on a single specified strain using the entire dataset.

    Parameters
    ----------
    base_path : str
        Project root path.
    strain : str
        Strain to fine-tune on. Must be a single strain name (e.g., "Escherichia coli").
    logger : CustomLogger
        Logging object.
    train_input_file : str
        Path to the training data CSV file (e.g., data/processed/{suffix}/All_Integrated_aggregated.csv).
    ft_epochs : int, optional
        Number of fine-tuning epochs (default: 20).
    device : str, optional
        Device to run training on (default: 'cuda:0').
    freeze_part : str, optional
        Part of the model to freeze. Must be one of: "inception", "transformer", "dense", or "none".
        - "inception": Freeze both inception1 and inception2 modules.
        - "transformer": Freeze the transformer_encoder module.
        - "dense": Freeze the dense layer.
        - "none": Do not freeze any layers.
        (default: "inception")
    patience : int, optional
        Number of epochs to wait for improvement in training loss before early stopping (default: 3).
    base_model_path : str, optional
        Path to the pretrained model checkpoint (.pt file). If not specified, defaults to
        'base_path/experiments/models/{suffix}/ania.pt'.
    output_path : str, optional
        Path to save the fine-tuned model. If not specified, defaults to
        'base_path/experiments/models/{suffix}/ania_finetuned.pt'.
    learning_rate : float, optional
        Learning rate for fine-tuning (default: 1e-5).
    weight_decay : float, optional
        Weight decay for regularization (default: 0.001).
    dropout_rate : float, optional
        Dropout rate for regularization (default: 0.5).
    batch_size : int, optional
        Batch size for training (default: 128).
    loss_threshold : float, optional
        Threshold for training loss change to trigger early stopping (default: 1e-4).

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

        # Validate the train_input_file
        if not os.path.exists(train_input_file):
            raise FileNotFoundError(
                f"Training data file not found at '{train_input_file}'"
            )

        # Set the strain to train on
        suffix = all_strains[strain]

        # Define the target column and metadata columns
        target_column = "Log MIC Value"
        metadata_columns = ["ID", "Sequence", "Targets"]

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

        # Log the strain being fine-tuned
        logger.add_divider(level=logging.INFO, length=80, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Strain - '{strain}' -> Model - 'ANIA' -> Feature - '{feature_set['name']}'",
            border="|",
            length=80,
        )
        logger.add_divider(level=logging.INFO, length=80, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)

        # Extract FCGR features from the specified training file
        _, X_train_fcgr, y_train = extract_cgr_features_and_target_for_dl(
            file_path=train_input_file,
            metadata_columns=metadata_columns,
            target_column=target_column,
            feature_start_idx=feature_set["start_idx"],
            feature_end_idx=feature_set["end_idx"],
            height=feature_set["height"],
            width=feature_set["width"],
        )

        # Define model input and output paths
        if base_model_path is None:
            base_model_path = os.path.join(
                base_path,
                f"experiments/models/{suffix}/ania.pt",
            )
        if output_path is None:
            output_path = os.path.join(
                base_path,
                f"experiments/models/{suffix}/ania_finetuned",
            )

        # Run fine-tuning for ANIA
        fine_tune_model(
            base_model_path=base_model_path,
            X_train_fcgr=X_train_fcgr,
            y_train=y_train,
            output_path=output_path,
            logger=logger,
            device=device,
            epochs=ft_epochs,
            freeze_part=freeze_part,
            batch_size=batch_size,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            loss_threshold=loss_threshold,
        )

        # Insert a blank line in the log for readability
        logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_fine_tune_pipeline()'.")
        raise


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script for ANIA model.")
    parser.add_argument(
        "--base_path", type=str, required=True, help="Project root path."
    )
    parser.add_argument(
        "--strain", type=str, required=True, help="Strain to fine-tune on."
    )
    parser.add_argument(
        "--train_input_file",
        type=str,
        required=True,
        help="Path to training data CSV file.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to save fine-tuned model."
    )
    parser.add_argument(
        "--log_path", type=str, required=True, help="Path to save log file."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run training on."
    )
    parser.add_argument(
        "--ft_epochs", type=int, default=20, help="Number of fine-tuning epochs."
    )
    parser.add_argument(
        "--freeze_part",
        type=str,
        default="inception",
        help="Part to freeze: 'inception', 'transformer', 'dense', or 'none'.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Patience for early stopping based on training loss.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay for regularization.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate for regularization.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--loss_threshold",
        type=float,
        default=1e-4,
        help="Threshold for training loss change to trigger early stopping.",
    )
    args = parser.parse_args()

    # Setup logger
    log_config_file = os.path.join(args.base_path, "configs/general_logging.json")
    logger = setup_logging(
        input_config_file=log_config_file,
        output_log_path=args.log_path,
        logger_name="general_logger",
        handler_name="general",
    )

    # Run fine-tuning pipeline
    run_fine_tune_pipeline(
        base_path=args.base_path,
        strain=args.strain,
        logger=logger,
        train_input_file=args.train_input_file,
        ft_epochs=args.ft_epochs,
        device=args.device,
        freeze_part=args.freeze_part,
        patience=args.patience,
        base_model_path=args.base_model_path,
        output_path=args.output_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        loss_threshold=args.loss_threshold,
    )


if __name__ == "__main__":
    main()
