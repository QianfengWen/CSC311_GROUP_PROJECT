import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
from tqdm import tqdm


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

class LogisticRegression(nn.Module):
    def __init__(self, num_questions):
        """Initialize a class LogisticRegression.
        
        Args:
            num_questions: int, number of questions in dataset
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_questions, num_questions)

    def forward(self, x):
        """Forward pass of logistic regression.
        
        Args:
            x: torch.Tensor of shape (batch_size, num_questions)
            
        Returns:
            out: torch.Tensor of shape (batch_size, 1), probability between 0 and 1
        """
        out = torch.sigmoid(self.linear(x))
        return out


def train(model, lr, train_data, zero_train_data, valid_data, num_epoch, device):
    """Train the logistic regression model."""
    model.train()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    valid_accs = []

    for epoch in tqdm(range(num_epoch)):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0).to(device)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            loss = F.binary_cross_entropy(output, target, reduction='sum') 

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, device)
        # tqdm.write(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f}\t Valid Acc: {valid_acc:.4f}")
        
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
    
    # report final validation accuracy
    tqdm.write('Final validation accuracy: {}'.format(valid_accs[-1]))


def evaluate(model, train_data, valid_data, device):
    """Evaluate the model on validation data.
    
    Args:
        model: LogisticRegression instance
        train_data: 2D FloatTensor, training data with 0s for missing values
        valid_data: dict, validation data
        
    Returns:
        float: accuracy on validation set
    """
    model.eval()
    model.to(device)
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = train_data[u].unsqueeze(0).to(device)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    """Main function to run logistic regression."""
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    model = LogisticRegression(train_matrix.shape[1]).to(device)
    
    # Set hyperparameters
    # lr = 0.001
    # lr = 0.01
    lr = 5e-5   
    num_epoch = 50
    
    # Train
    train(model, lr, train_matrix, zero_train_matrix, valid_data, num_epoch, device)
    
    # Evaluate on test set
    test_acc = evaluate(model, zero_train_matrix, test_data, device)
    tqdm.write(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
