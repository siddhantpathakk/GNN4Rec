from main import UltraGCN, params, train, test, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, constraint_mat, ii_constraint_mat, ii_neighbor_mat
import torch
import optuna
import matplotlib.pyplot as plt

def objective(trial):
    # Define the search space for lambda and gamma
    lambda_ = trial.suggest_loguniform("lambda", 5e-4, 1e1)
    gamma = trial.suggest_loguniform("gamma", 1e-4, 1e1)

    # Set the lambda and gamma values in your model's parameters
    params['lambda'] = lambda_
    params['gamma'] = gamma
    
    # Create and train the UltraGCN model
    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    ultragcn = ultragcn.to(params['device'])
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])
    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)

    # Evaluate the model and return the metric to optimize (e.g., validation loss)
    F1_score, _, _, _ = test(ultragcn, test_loader, test_ground_truth_list, mask, params['topk'], params['user_num'])
    
    # Store the F1-score and hyperparameter values in the study object
    trial.set_user_attr("F1_score", F1_score)
    trial.set_user_attr("lambda", lambda_)
    trial.set_user_attr("gamma", gamma)
    
    return -F1_score  # Optuna minimizes the objective, so we negate F1-score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # Use "maximize" for F1-score
    study.optimize(objective, n_trials=20)  # You can adjust the number of trials

    # Retrieve results
    results = study.trials_dataframe()

    # Plot F1-score vs. trial number
    plt.figure(figsize=(10, 6))
    plt.plot(results["number"], results["user_attrs/F1_score"], marker="o", linestyle="-", color="b")
    plt.xlabel("Trial Number")
    plt.ylabel("F1-score")
    plt.title("F1-score vs. Trial Number")
    plt.grid(True)
    plt.savefig('f1_score.png')

    # Plot lambda vs. F1-score
    plt.figure(figsize=(10, 6))
    plt.scatter(results["user_attrs/lambda"], results["user_attrs/F1_score"], c=results["number"], cmap="viridis")
    plt.xlabel("Lambda")
    plt.ylabel("F1-score")
    plt.title("F1-score vs. Lambda")
    plt.colorbar(label="Trial Number")
    plt.grid(True)
    plt.savefig('lambda.png')

    # Plot gamma vs. F1-score
    plt.figure(figsize=(10, 6))
    plt.scatter(results["user_attrs/gamma"], results["user_attrs/F1_score"], c=results["number"], cmap="viridis")
    plt.xlabel("Gamma")
    plt.ylabel("F1-score")
    plt.title("F1-score vs. Gamma")
    plt.colorbar(label="Trial Number")
    plt.grid(True)
    plt.savefig('gamma.png')

    # Print the best hyperparameters and result
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))