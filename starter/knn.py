import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Transpose the matrix to perform item-based filtering
    matrix_t = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    # Fit and transform on transposed matrix
    mat = nbrs.fit_transform(matrix_t)
    # Transpose back to original shape
    mat = mat.T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Define k values to test
    k_values = [1, 6, 11, 16, 21, 26]
    
    # User-based collaborative filtering
    print("\nUser-based collaborative filtering:")
    user_accuracies = []
    for k in k_values:
        print(f"\nk = {k}")
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_accuracies.append(acc)
    
    # Plot user-based results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, user_accuracies, 'bo-')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('User-based Collaborative Filtering: Accuracy vs k')
    plt.grid(True)
    plt.show()
    
    # Find best k and evaluate on test set
    best_k_user = k_values[np.argmax(user_accuracies)]
    print(f"Best k for user-based: {best_k_user}")
    final_user_acc = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    print(f"Final test accuracy (user-based): {final_user_acc}")
    
    # Item-based collaborative filtering
    print("\nItem-based collaborative filtering:")
    item_accuracies = []
    for k in k_values:
        print(f"\nk = {k}")
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_accuracies.append(acc)
    
    # Plot item-based results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, item_accuracies, 'ro-')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Item-based Collaborative Filtering: Accuracy vs k')
    plt.grid(True)
    plt.show()
    
    # Find best k and evaluate on test set
    best_k_item = k_values[np.argmax(item_accuracies)]
    print(f"Best k for item-based: {best_k_item}")
    final_item_acc = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print(f"Final test accuracy (item-based): {final_item_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
