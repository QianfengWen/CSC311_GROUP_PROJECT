# TODO: complete this file.

from item_response import *
import numpy as np


def create_bootstrap_samples(original_data, num_samples):
    """Generate bootstrap samples from the original dataset.
    
    Args:
        original_data (dict): Original training data dictionary
        num_samples (int): Number of bootstrap samples to create
        
    Returns:
        list: List of bootstrap sample dictionaries
    """
    bootstrap_samples = []
    data_size = len(original_data["user_id"])
    
    for _ in range(num_samples):
        # Sample with replacement
        indices = np.random.randint(0, data_size, size=data_size)
        
        # Create new sample dictionary
        sample = {
            "user_id": np.array(original_data["user_id"])[indices],
            "question_id": np.array(original_data["question_id"])[indices], 
            "is_correct": np.array(original_data["is_correct"])[indices]
        }
        bootstrap_samples.append(sample)
        
    return bootstrap_samples


def evaluate_ensemble(test_data, theta_list, beta_list, ensemble_size):
    """Evaluate ensemble predictions using majority voting.
    
    Args:
        test_data (dict): Test dataset
        theta_list (list): List of theta parameters from each model
        beta_list (list): List of beta parameters from each model
        ensemble_size (int): Number of models in ensemble
        
    Returns:
        float: Accuracy of ensemble predictions
    """
    # Get predictions from each model
    all_predictions = np.zeros((ensemble_size, len(test_data["question_id"])))
    
    for model_idx in range(ensemble_size):
        predictions = []
        for i, question in enumerate(test_data["question_id"]):
            user = test_data["user_id"][i]
            logit = (theta_list[model_idx][user] - beta_list[model_idx][question]).sum()
            prob = sigmoid(logit)
            predictions.append(prob >= 0.5)
        all_predictions[model_idx] = np.array(predictions)
    
    # Combine predictions using majority voting
    ensemble_predictions = np.zeros(len(test_data["question_id"]))
    ensemble_predictions[np.mean(all_predictions, axis=0) > 0.5] = 1
    
    # Calculate accuracy
    accuracy = np.mean(test_data["is_correct"] == ensemble_predictions)
    return accuracy


def main():
    # Load data
    train_data = load_train_csv("./data")
    validation_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data") 

    # Set hyperparameters
    learning_rate = 0.01
    iterations = 50
    num_bootstrap_samples = 3

    # Generate bootstrap samples
    bootstrap_samples = create_bootstrap_samples(train_data, num_bootstrap_samples)
    
    # Lists to store model parameters and metrics
    theta_parameters = []
    beta_parameters = []
    validation_accuracies = []
    test_accuracies = []

    # Train individual models on bootstrap samples
    for sample in bootstrap_samples:
        theta, beta, val_acc_history, val_nll = irt(
            sample, validation_data, learning_rate, iterations
        )
        theta_parameters.append(theta)
        beta_parameters.append(beta)
        
        # Evaluate individual model performance
        val_acc = evaluate(validation_data, theta, beta)
        test_acc = evaluate(test_data, theta, beta)
        validation_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

    # Evaluate ensemble performance
    ensemble_val_acc = evaluate_ensemble(
        validation_data, theta_parameters, beta_parameters, num_bootstrap_samples
    )
    ensemble_test_acc = evaluate_ensemble(
        test_data, theta_parameters, beta_parameters, num_bootstrap_samples
    )
    validation_accuracies.append(ensemble_val_acc)
    test_accuracies.append(ensemble_test_acc)

    # Train and evaluate on original dataset
    theta, beta, val_acc_history, val_nll = irt(
        train_data, validation_data, learning_rate, iterations
    )
    original_val_acc = evaluate(validation_data, theta, beta)
    original_test_acc = evaluate(test_data, theta, beta)
    validation_accuracies.append(original_val_acc)
    test_accuracies.append(original_test_acc)

    # Print results
    print("RESULTS SUMMARY")
    print('='*60)
    print('Result for IRT without bagging:')
    print('Final validation accuracy: {}'.format(validation_accuracies[0]))
    print('Final test accuracy: {}'.format(test_accuracies[0]))
    print('='*60)
    print('Final validation accuracy: {}'.format(validation_accuracies[1]))
    print('Final test accuracy: {}'.format(test_accuracies[1]))
    print('='*60)
    print('Final validation accuracy: {}'.format(validation_accuracies[2]))
    print('Final test accuracy: {}'.format(test_accuracies[2]))
    print('='*60)
    print('Result for IRT with bagging:')
    print('Final ensemble validation accuracy: {}'.format(validation_accuracies[3]))
    print('Final ensemble test accuracy: {}'.format(test_accuracies[3]))
    print('='*60)
    print('Result for IRT original training data :')
    print('Final validation accuracy: {}'.format(validation_accuracies[4]))
    print('Final test accuracy: {}'.format(test_accuracies[4]))



if __name__ == "__main__":
    main()