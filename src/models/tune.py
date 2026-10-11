import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Silencing minor optuna alerts 
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_model(df: pd.DataFrame, target_column:str) -> dict:
    """
        Hyperparameter tuning using Optuna

        Arguments:
            df(pd.DataFrame): Encoded feature engineered dataset.
            target_column(str): Name of target variable("Churn).

        Returns:
            dict: best hyperparameters.
    """

    x = df.drop(columns=[target_column])
    y = df[target_column]

    # Feature scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    def objective(trial):
        # setting the rule num_leaves < 2^max_depth
        max_depth = trial.suggest_int("max_depth", 3, 8)
        max_leaves_allowed = min(64, (2**max_depth) - 1)
        num_leaves = trial.suggest_int("num_leaves", 7, max_leaves_allowed)
        # parameters
        params= {
            # binary classification problem
            "objective" : "binary",
            # evaluation metrics optimised to trace binary probability errors
            "metric" : "binary_logloss",
            # gradient boosting decision tree
            "boosting_type" : "gbdt",
            # ensure reproducibility
            "random_state" : 42,
            # utilises all available cores in parallel
            "n_jobs" : -1,
            # silences logs
            "verbose" : -1,
            # restricts tree depth
            "max_depth" : max_depth,
            # restricts the maximum layout
            "num_leaves" : num_leaves,
            #  penalty for missing out a churner using Bayesian engine
            "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 1.5, 4.0),
            # step size rate
            "learning_rate" : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # total trees built
            "n_estimators" : trial.suggest_int("n_estimators", 300, 800, step=50),
            # row sample percentage
            "subsample" : trial.suggest_float("subsample", 0.5, 1.0),
            # enables row sampling
            "bagging_freq" : 1,
            # column sample percentage
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # L1 regularisation
            "reg_alpha" : trial.suggest_float("reg_alpha", 0.001, 2.0, log=True),
            # L2 regularisation
            "reg_lambda" : trial.suggest_float("reg_lambda", 0.001, 2.0, log=True)
        }
        # final model - LightGBM
        model = LGBMClassifier(**params)
        # cross validation
        cv_score_LGBM = cross_val_score(
            model, x_scaled, y, cv = 5, scoring="recall", n_jobs= 1
        )
        # returns the mean of all 5 recall score of each folds
        return cv_score_LGBM.mean()

    print("Starting Hyperparameter tuning using Optuna......")

    # creating study and optimising the hyperparameters in 30 trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print(f"Optimisation Complete. Best Recall: {study.best_value:.4f}")
    print(f"Best Hyperparameters: {study.best_params}")

    return study.best_params




