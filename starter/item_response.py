from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # data (theta - beta) - log (1 + exp(theta - beta))
    log_lklihood = np.sum(data["is_correct"] * (theta[data["user_id"]] - beta[data["question_id"]]) - np.log(1 + np.exp(theta[data["user_id"]] - beta[data["question_id"]])))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # sigmoid(theta - beta)
    sig = sigmoid(theta[data["user_id"]] - beta[data["question_id"]])
    
    # d/dtheta = c - sigmoid(theta - beta) 
    theta_grad = data["is_correct"] - sig
     
    # d/dbeta = sigmoid(theta - beta) - c
    beta_grad = sig - data["is_correct"]
    
    theta_update = np.zeros(len(theta))
    beta_update = np.zeros(len(beta))
    
    np.add.at(theta_update, data["user_id"], theta_grad)
    
    np.add.at(beta_update, data["question_id"], beta_grad)
    
    theta = theta + (lr * theta_update)
    beta = beta + (lr * beta_update)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    num_users = max(data["user_id"]) + 1
    num_questions = max(data["question_id"]) + 1
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    train_nll_lst = []
    val_nll_lst = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        # val_acc_lst.append(score)
        train_nll_lst.append(neg_lld)
        val_nll_lst.append(val_neg_lld)
        print(f"iteration {i+1}/{iterations} - train NLL: {neg_lld:.4f} \t validation NLL: {val_neg_lld:.4f}")
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_nll_lst, val_nll_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def plot_probability_curves(theta, beta, selected_questions, save_path='probability_curves.png'):
    plt.figure(figsize=(10, 6))
    theta_range = np.linspace(min(theta) - 1, max(theta) + 1, 300)

    for j in selected_questions:
        beta_j = beta[j]
        p_correct = sigmoid(theta_range - beta_j)
        plt.plot(theta_range, p_correct, label=f'problem {j} (β={beta_j:.2f})')

    plt.xlabel('θ (student ability)')
    plt.ylabel('p(cij = 1)')
    plt.title('Different problems with student ability')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def main():
    train_data = load_train_csv("/Users/quanjunwei/PycharmProjects/CSC311_GROUP_PROJECT/starter/data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("/Users/quanjunwei/PycharmProjects/CSC311_GROUP_PROJECT/starter/data")
    test_data = load_public_test_csv("/Users/quanjunwei/PycharmProjects/CSC311_GROUP_PROJECT/starter/data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 100
    theta, beta, train_nll, val_nll = irt(train_data, val_data, lr, iterations)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations + 1), train_nll, label='train negative log likelihood')
    plt.plot(range(1, iterations + 1), val_nll, label='validation negative log likelihood')
    plt.xlabel('iterations')
    plt.ylabel('negative log likelihood')
    plt.title('training and validation negative log likelihood curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.show()

    test_nll = neg_log_likelihood(test_data, theta=theta, beta=beta)
    test_accuracy = evaluate(test_data, theta=theta, beta=beta)
    print("test negative log likelihood: {:.4f}".format(test_nll))
    print("test accuracy: {:.4f}".format(test_accuracy))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    sorted_questions = np.argsort(beta)
    j1 = sorted_questions[0]  # easiest
    j2 = sorted_questions[len(beta) // 2]  # medium
    j3 = sorted_questions[-1]  # hardest

    selected_questions = [j1, j2, j3]
    plot_probability_curves(theta, beta, selected_questions, save_path='probability_curves.png')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
