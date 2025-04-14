# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-nested-blocks
"""
Deep Learning Model Testing Module for ANIA

This module provides functionality for testing the trained ANIA deep learning model
on AMP-MIC data to predict Minimum Inhibitory Concentration (MIC). It evaluates model performance
using standard regression metrics and supports structured logging.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time
from typing import Optional

# ============================== Third-Party Library Imports ==============================
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.amp import GradScaler, autocast  # type: ignore

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger

# Deep Learning Utility Functions
from models.deep_learning.ANIA import ANIA
from src.models.deep_learning.utils import extract_cgr_features_and_target_for_dl


# ============================== Custom Function ==============================
def check_cuda_and_optimize(device: str, logger: CustomLogger) -> tuple[list[int], str]:
    """
    Check CUDA availability and set up optimizations for GPU testing if available.

    Parameters
    ----------
    device : str
        Device to run the model on (e.g., 'cuda', 'cuda:0', 'cuda:0,1', or 'cpu').
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    tuple[list[int], str]
        A tuple containing:
        - List of GPU IDs to use (e.g., [0, 1], or [] if using CPU).
        - Primary device (e.g., 'cuda:0' or 'cpu').
    """
    # Initialize return values
    device_ids = []
    primary_device = device.lower()

    # If device is specified as CPU, return immediately
    if "cpu" in primary_device:
        logger.log_with_borders(
            level=logging.INFO,
            message="Using 'CPU' device as specified.",
            border="|",
            length=100,
        )
        return device_ids, "cpu"

    # Check CUDA availability
    if "cuda" in primary_device and torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Number of 'available' GPUs: {num_gpus}",
            border="|",
            length=100,
        )

        # Parse the device string
        if primary_device == "cuda":
            # Use all available GPUs
            device_ids = list(range(num_gpus))
            primary_device = "cuda:0"
        elif "," in primary_device:
            # Use specified GPUs (e.g., 'cuda:0,1')
            try:
                device_ids = [
                    int(idx) for idx in primary_device.replace("cuda:", "").split(",")
                ]
                primary_device = f"cuda:{device_ids[0]}"
            except ValueError:
                raise ValueError(
                    f"Invalid device format: '{device}'. Expected format: 'cuda:<index1>,<index2>' (e.g., 'cuda:0,1')."
                )
        else:
            # Use a single specified GPU (e.g., 'cuda:0')
            try:
                device_idx = (
                    int(primary_device.split(":")[1]) if ":" in primary_device else 0
                )
                device_ids = [device_idx]
                primary_device = primary_device
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
    else:
        # If CUDA is not available or not specified, fall back to CPU
        logger.log_with_borders(
            level=logging.INFO,
            message="CUDA is not available or not specified, using 'CPU' device.",
            border="|",
            length=100,
        )
        primary_device = "cpu"

    return device_ids, primary_device


def evaluate_dl_predictions(
    predictions_csv_path: str,
    logger: CustomLogger,
) -> None:
    """
    Compute evaluation metrics from saved prediction CSV.

    Parameters
    ----------
    predictions_csv_path : str
        Path to the CSV file containing actual and predicted values.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load predictions
        df = pd.read_csv(predictions_csv_path)

        # Ensure required columns exist
        if (
            "Log MIC Value" not in df.columns
            or "Predicted Log MIC Value" not in df.columns
        ):
            raise KeyError("Missing required columns in prediction CSV.")

        # Extract actual and predicted values
        y_true = df["Log MIC Value"]
        y_pred = df["Predicted Log MIC Value"]

        # Compute evaluation metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        pcc = np.corrcoef(y_true, y_pred)[0, 1]  # Pearson Correlation Coefficient

        # Log results
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Evaluation Metrics for 'ANIA' model:\n"
                f"  • Mean Absolute Error (MAE): {mae:.4f}\n"
                f"  • Mean Squared Error (MSE): {mse:.4f}\n"
                f"  • Root Mean Squared Error (RMSE): {rmse:.4f}\n"
                f"  • R² Score: {r2:.4f}\n"
                f"  • Pearson Correlation Coefficient (PCC): {pcc:.4f}"
            ),
            border="|",
            length=100,
        )

    except FileNotFoundError:
        logger.exception(msg="FileNotFoundError in 'evaluate_dl_predictions()'.")
        raise

    except KeyError:
        logger.exception(msg="KeyError in 'evaluate_dl_predictions()'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'evaluate_dl_predictions()'.")
        raise


def load_trained_ania_model(
    checkpoint_path: str,
    device: str,
    logger: CustomLogger,
) -> nn.Module:
    """
    Load a trained ANIA model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint file.
    device : str
        Device to load the model onto (e.g., 'cuda:0' or 'cpu').
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    nn.Module
        Loaded ANIA model instance ready for inference.
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract hyperparameters used during training
        hyperparams = checkpoint.get("hyperparams")
        if hyperparams is None:
            raise ValueError("Missing 'hyperparams' in checkpoint file.")

        # Dynamically compute inception1_out_channels and inception2_out_channels
        inception1_out_channels = (
            hyperparams["inception1_branch1x1_channels"]
            + hyperparams["inception1_branch3x3_channels"]
            + hyperparams["inception1_branch5x5_channels"]
            + hyperparams["inception1_branch_pool_channels"]
        )
        inception2_out_channels = (
            hyperparams["inception2_branch1x1_channels"]
            + hyperparams["inception2_branch3x3_channels"]
            + hyperparams["inception2_branch5x5_channels"]
            + hyperparams["inception2_branch_pool_channels"]
        )

        # Initialize ANIA model
        model = ANIA(
            in_channels=11,
            inception1_out_channels=inception1_out_channels,
            inception2_out_channels=inception2_out_channels,
            inception1_branch1x1_channels=hyperparams["inception1_branch1x1_channels"],
            inception1_branch3x3_channels=hyperparams["inception1_branch3x3_channels"],
            inception1_branch3x3_reduction=hyperparams[
                "inception1_branch3x3_reduction"
            ],
            inception1_branch5x5_channels=hyperparams["inception1_branch5x5_channels"],
            inception1_branch5x5_reduction=hyperparams[
                "inception1_branch5x5_reduction"
            ],
            inception1_branch_pool_channels=hyperparams[
                "inception1_branch_pool_channels"
            ],
            inception2_branch1x1_channels=hyperparams["inception2_branch1x1_channels"],
            inception2_branch3x3_channels=hyperparams["inception2_branch3x3_channels"],
            inception2_branch3x3_reduction=hyperparams[
                "inception2_branch3x3_reduction"
            ],
            inception2_branch5x5_channels=hyperparams["inception2_branch5x5_channels"],
            inception2_branch5x5_reduction=hyperparams[
                "inception2_branch5x5_reduction"
            ],
            inception2_branch_pool_channels=hyperparams[
                "inception2_branch_pool_channels"
            ],
            num_heads=hyperparams["num_heads"],
            d_model=hyperparams["d_model"],
            num_encoder_layers=hyperparams.get("num_encoder_layers", 2),
            dense_hidden_dim=hyperparams["dense_hidden_dim"],
            dropout_rate=hyperparams["dropout_rate"],
        ).to(device)

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully loaded trained 'ANIA' model from checkpoint:\n'{checkpoint_path}'",
            border="|",
            length=100,
        )
        return model

    except Exception:
        logger.exception(msg=f"Unexpected error in 'load_trained_ania_model()'.")
        raise


def test_dl_model(
    df_metadata: pd.DataFrame,
    X_test_fcgr: torch.Tensor,
    y_test: torch.Tensor,
    model_input_path: str,
    prediction_output_path: str,
    logger: CustomLogger,
    device: str = "cuda:0",
) -> None:
    """
    Test a trained ANIA model on test data and save predictions.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        DataFrame containing metadata (e.g., ID, Sequence, Targets).
    X_test_fcgr : torch.Tensor
        FCGR test feature tensor of shape (batch_size, in_channels, height, width) for 'ANIA'.
    y_test : torch.Tensor
        Test target tensor of shape (batch_size, 1).
    model_input_path : str
        Path to the trained model checkpoint (.pt file).
    prediction_output_path : str
        Path to save the prediction results as CSV.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    device : str, optional
        Device to run the model on (e.g., 'cuda:0' or 'cpu', default: 'cuda:0').

    Returns
    -------
    None
    """
    try:
        # Log testing start
        logger.info(msg=f"/ Task: Test 'ANIA' model")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record start time
        start_time = time.time()

        # Check device and set up optimizations
        device_ids, primary_device = check_cuda_and_optimize(device, logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Load model structure and state_dict
        model = load_trained_ania_model(
            checkpoint_path=model_input_path,
            device=primary_device,
            logger=logger,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Move test data to device
        X_test_fcgr = X_test_fcgr.to(primary_device)
        y_test = y_test.to(primary_device)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Moved testing data to '{primary_device}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Predict with mixed precision
        with torch.no_grad():
            with autocast("cuda"):
                y_pred = model(X_test_fcgr)
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_test.cpu().numpy().flatten()

        # Save predictions
        df_results = df_metadata.copy()
        df_results["Log MIC Value"] = y_true
        df_results["Predicted Log MIC Value"] = y_pred

        os.makedirs(os.path.dirname(prediction_output_path), exist_ok=True)
        df_results.to_csv(prediction_output_path, index=False)

        # Compute evaluation metrics
        evaluate_dl_predictions(prediction_output_path, logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record end time
        total_time = time.time() - start_time
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Testing completed for 'ANIA' model:\n"
            f"  • Time: {total_time:.2f}s\n"
            f"  • Running on device: {primary_device}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log successful save
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Predictions saved:\n'{prediction_output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except ValueError:
        logger.exception(msg="ValueError in 'test_dl_model()' for 'ANIA'.")
        raise

    except FileNotFoundError:
        logger.exception(msg="FileNotFoundError in 'test_dl_model()' for 'ANIA'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'test_dl_model()' for 'ANIA'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_test_dl_pipeline(
    base_path: str,
    strain: str,
    logger: CustomLogger,
    device: str = "cuda:0",
    model_input_path: Optional[str] = None,
    test_input_file: Optional[str] = None,
    prediction_output_path: Optional[str] = None,
) -> None:
    """
    Run the deep learning testing pipeline for ANIA model on a single specified strain.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    strain : str
        Strain to test on. Must be a single strain name (e.g., "Escherichia coli").
    logger : CustomLogger
        Logger for structured logging throughout the pipeline.
    device : str, optional
        Device to run the model on (default: 'cuda:0').
    model_input_path : str, optional
        Path to the trained model checkpoint (.pt file).
    test_input_file : str, optional
        Path to the test data CSV file.
    prediction_output_path : str, optional
        Path to save the prediction results as CSV.

    Returns
    -------
    None
        This function performs file I/O and logging side effects, and does not return any value.
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

        # Log the strain being tested
        logger.add_divider(level=logging.INFO, length=80, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Strain - '{strain}' -> Model - 'ANIA' -> Feature - '{feature_set['name']}'",
            border="|",
            length=80,
        )
        logger.add_divider(level=logging.INFO, length=80, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)

        # Extract metadata, features, and target variable for test set
        if test_input_file is None:
            test_input_file = os.path.join(
                base_path, f"data/processed/split/{suffix}_test.csv"
            )
        df_metadata, X_test_fcgr, y_test = extract_cgr_features_and_target_for_dl(
            file_path=test_input_file,
            metadata_columns=metadata_columns,
            target_column=target_column,
            feature_start_idx=feature_set["start_idx"],
            feature_end_idx=feature_set["end_idx"],
            height=feature_set["height"],
            width=feature_set["width"],
        )

        # Define model input and prediction output paths
        if model_input_path is None:
            model_input_path = os.path.join(
                base_path,
                f"experiments/models/{suffix}/ania.pt",
            )
        if prediction_output_path is None:
            prediction_output_path = os.path.join(
                base_path,
                f"experiments/predictions/{suffix}/deep_learning/test_predict.csv",
            )

        # Test the ANIA model on test set
        test_dl_model(
            df_metadata=df_metadata,
            X_test_fcgr=X_test_fcgr,
            y_test=y_test,
            model_input_path=model_input_path,
            prediction_output_path=prediction_output_path,
            logger=logger,
            device=device,
        )

        # Insert a blank line in the log for readability
        logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_test_dl_pipeline()'.")
        raise
