import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y, num_epochs=5000, learning_rate=0.01, tolerance=1e-5):
    """
    Train the model for the given number of epochs.
    """
    input_features = X.shape[1]
    output_features = y.shape[1]
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Use mean squared error loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        # Print the loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Stop training if the loss does not change significantly
        if abs(previous_loss - loss.item()) < tolerance:
            break

        previous_loss = loss.item()

    return model, loss

# Normalizing Data Function
def normalize_data(X, y):
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)
    X = (X - X_mean) / X_std

    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True)
    y = (y - y_mean) / y_std

    return X, y, X_mean, X_std, y_mean, y_std

def denormalize_data(y, y_mean, y_std):
    return y * y_std + y_mean