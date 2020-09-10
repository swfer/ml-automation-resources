def shift_columns(matrix, shifts_number, upwards=True):
    # Copy the dataframe's schema without any column
    ar_matrix = matrix[[]].copy()
    
    # Add shifted columns for each column
    for column in matrix.columns:
        for shift_index in range(1, shifts_number + 1):
            shift_index_signed = -shift_index if upwards else shift_index
            ar_matrix[column + ('+' if upwards else '-') + str(shift_index)] = matrix[column].shift(shift_index_signed)

    return ar_matrix

def create_autoregressive_dataset(X, Y, lags_number=1, horizon_number=1, remove_nans=True):
    # Create negative shifts for each dataset's features
    ar_dataset_X = X.copy().join(shift_columns(X, shifts_number=lags_number, upwards=False))
    
    # Create positive shifts for each dataset's targets
    ar_dataset_Y = shift_columns(Y, shifts_number=horizon_number)

    # Join X and Y for nan's removal
    if(remove_nans):
        dataset_na_free = ar_dataset_X.copy().join(ar_dataset_Y).dropna()
        ar_dataset_X = dataset_na_free[ar_dataset_X.columns]
        ar_dataset_Y = dataset_na_free[ar_dataset_Y.columns]

    # Return new X and Y
    print(ar_dataset_X, ar_dataset_Y)
    return ar_dataset_X, ar_dataset_Y