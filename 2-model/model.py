import wandb

# Weights and Biases Setup
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="zontasticality-university-of-massachusetts-amherst",
    # Set the wandb project where this run will be logged.
    project="ping-model",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# To use
# run.log({"acc": acc, "loss": loss})
# run.finish()

