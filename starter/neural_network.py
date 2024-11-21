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


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        hidden = torch.sigmoid(self.g(inputs))
        output = torch.sigmoid(self.h(hidden))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return output


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_loss_history = []
    valid_acc_history = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            nan_mask = torch.from_numpy(nan_mask).bool()
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            if lamb != 0:
                loss += lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )
        train_loss_history.append(train_loss)
        valid_acc_history.append(valid_acc)
    return train_loss_history, valid_acc_history
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data("/Users/quanjunwei/PycharmProjects/CSC311_GROUP_PROJECT/starter/data")

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = [10, 50, 100, 200, 500]
    best_k = None
    best_acc = 0
    results = {}
    model = None

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 50
    lamb = 0

    for k in k:
        print(f"Training with k={k}")
        model = AutoEncoder(train_matrix.shape[1], k)
        train_loss_history, valid_acc_history = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        final_acc = valid_acc_history[-1]
        results[k] = final_acc
        if final_acc > best_acc:
            best_acc = final_acc
            best_k = k
            best_train_loss_history = train_loss_history
            best_valid_acc_history = valid_acc_history
    print(f"Best k selected: {best_k} with validation accuracy: {best_acc}")

    epochs = range(1, num_epoch + 1)
    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    line1 = ax1.plot(epochs, best_train_loss_history, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create second y-axis that shares x-axis
    ax2 = ax1.twinx()

    # Plot validation accuracy on right y-axis
    color = 'tab:red'
    ax2.set_ylabel('Validation Accuracy', color=color)
    line2 = ax2.plot(epochs, best_valid_acc_history, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title
    plt.title(f'Training Loss and Validation Accuracy vs. Epoch (k={best_k})')

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    # Add grid
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_validation_metrics_k{}.png'.format(best_k))
    plt.show()


    # Next, evaluate your network on validation/test data

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Test Accuracy with k = {best_k}: {test_acc}")

    # add l2 regularization
    lambda_values = [0, 0.001, 0.01, 0.1, 1]
    best_lambda = None
    best_reg_acc = 0
    best_reg_train_acc = 0
    best_reg_valid_acc = 0

    print("\nTraining with L2 Regularization")
    for lamb in lambda_values:
        print(f"\nTraining with k={best_k} and lambda={lamb}")
        model = AutoEncoder(num_question=zero_train_matrix.shape[1], k=best_k)
        train_loss_history, valid_acc_history = train(
            model,
            lr=lr,
            lamb=lamb,
            train_data=train_matrix,
            zero_train_data=zero_train_matrix,
            valid_data=valid_data,
            num_epoch=num_epoch,
        )
        final_acc = valid_acc_history[-1]
        print(f"Validation Accuracy with lambda={lamb}: {final_acc}")
        if final_acc > best_reg_acc:
            best_reg_acc = final_acc
            best_lambda = lamb
            best_reg_train_loss = train_loss_history
            best_reg_valid_acc = valid_acc_history

    print(f"\nBest lambda selected: {best_lambda} with validation accuracy: {best_reg_acc}")

        # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss with Regularization', color=color)
    line1 = ax1.plot(epochs, best_reg_train_loss, color=color, label='Training Loss with Regularization')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create second y-axis that shares x-axis
    ax2 = ax1.twinx()

    # Plot validation accuracy on right y-axis
    color = 'tab:red'
    ax2.set_ylabel('Validation Accuracy', color=color)
    line2 = ax2.plot(epochs, best_reg_valid_acc, color=color, label='Validation Accuracy with Regularization')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title
    plt.title(f'Training and Validation Metrics for k={best_k}, λ={best_lambda}')

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    # Add grid
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_validation_metrics_k{}_lambda{}.png'.format(best_k, best_lambda))
    plt.show()

    test_acc_reg = evaluate(model, zero_train_matrix, test_data)
    print(f"Test Accuracy with k = {best_k} and lambda = {best_lambda}: {test_acc_reg}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
