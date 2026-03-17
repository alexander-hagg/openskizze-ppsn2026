# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Shared utility functions for Gaussian Process models.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

def train_gp(X_train, y_train, training_iter=2000):
    """
    Trains a Gaussian Process regressor with input standardization.
    
    Args:
        X_train: Training inputs (n_samples, n_features)
                 Expected: [heightmap_flat (l*l), slope_angle, slope_direction, urban_fraction]
        y_train: Training targets (n_samples,)
        training_iter: Unused (kept for API compatibility)
    
    Returns:
        Dictionary with 'model' and 'scaler' for predictions
    """
    # Standardize inputs (zero mean, unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train GP with scaled inputs
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gaussian_process.fit(X_train_scaled, y_train)
    
    # Return model and scaler together
    return {'model': gaussian_process, 'scaler': scaler}

def eval_gp(model_dict, X_test, return_std=True):
    """
    Evaluates a GP model, returning mean and optionally standard deviation.
    
    Args:
        model_dict: Dictionary with 'model' and 'scaler'
        X_test: Test inputs (n_samples, n_features)
        return_std: Whether to return standard deviation
    """
    X_test_scaled = model_dict['scaler'].transform(X_test)
    return model_dict['model'].predict(X_test_scaled, return_std=return_std)

def acquire_ucb(model_dict, X_test, lambda_ucb=0.1):
    """Calculates the Upper Confidence Bound (UCB) for a given set of test points."""
    mean_test, std_prediction = eval_gp(model_dict, X_test, return_std=True)
    return mean_test + lambda_ucb * std_prediction