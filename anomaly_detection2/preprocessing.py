import torch
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler


def sequence_data(train_data: npt.NDArray[any], test_data: npt.NDArray[any], sliding_window_len: int = 10,
                   sliding_window_step: int = 1) -> tuple[npt.NDArray[any], npt.NDArray[any]]: 

    X_train_tensor = torch.tensor(train_data)
    X_train_tensor = X_train_tensor.reshape(X_train_tensor.shape[0],X_train_tensor.shape[2], -1)
    X_train_tensor = X_train_tensor.unfold(0,sliding_window_len,sliding_window_step).squeeze().permute(0,2,1)

    X_test_tensor = torch.tensor(test_data)
    X_test_tensor = X_test_tensor.reshape(X_test_tensor.shape[0],X_test_tensor.shape[2], -1)
    X_test_tensor = X_test_tensor.unfold(0,sliding_window_len,sliding_window_step).squeeze().permute(0,2,1)

    train_data = X_train_tensor.numpy()
    test_data = X_test_tensor.numpy()

    return train_data, test_data

def normalize(train_data: torch.Tensor, test_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Normalize data
    scaler = MinMaxScaler()
    train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
    test_data_reshaped = test_data.reshape(-1, test_data.shape[-1])

    train_data_normalized = scaler.fit_transform(train_data_reshaped)
    test_data_normalized = scaler.transform(test_data_reshaped)

    train_data = train_data_normalized.reshape(train_data.shape)
    test_data = test_data_normalized.reshape(test_data.shape)

    # Convert data to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    return train_data, test_data