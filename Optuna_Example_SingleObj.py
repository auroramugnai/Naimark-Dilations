import optuna

def constraints(trial):
    """Function used to include constraints in the optimization with Optuna"""
    return trial.user_attrs["constraint"]

def objective(trial):
    """
    Function that defines the objective/s of the optimization for Optuna. It also includes
    the definitions of the problem constraints.
    """
    
    # We define some variables as random floats "x" and "y", and we define their ranges
    x = trial.suggest_float("x", 0, 5) # "x" is a random float from 0 to 5
    y = trial.suggest_float("y", 0, 3) # "y" is a random float from 0 to 3

    v0 = x**2 + y**2 # The objective is min(x^2 + y^2)

    # Constraints which are considered feasible if less than or equal to zero.
    c0 = - v0 + 1 # The constraint is that min(x^2 + y^2) >= 1
    
    trial.set_user_attr("constraint", (c0,c0)) # We add the constraints to the problem
    
    return v0

# Define sampler that includes the constraints
sampler = optuna.samplers.TPESampler(constraints_func=constraints)
pruner = optuna.pruners.HyperbandPruner()

# We define a study for the optimization
study = optuna.create_study(directions=["minimize"], sampler = sampler, pruner = pruner) 
# We set the objective and the number of trials (nº of iterations)
study.optimize(objective, n_trials=100)

# Since the constraints are not explicitly enforce, we want to post-select only the feasible trials
# that fulfill them, so we create a new study for this
new_study = optuna.create_study(directions=study.directions)

# We extract the feasible trials from our previous study
feasible_trials = []
for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
    if all(map(lambda x: x <= 0.0, constraints(trial))):
        feasible_trials.append(trial)
        
# And we add them to this new study
new_study.add_trials(feasible_trials)

# We print the best result and the corresponding parameters
print('Best value:', new_study.best_value, 'Best params:', new_study.best_params,  '\n')
