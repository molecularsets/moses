import argparse
from runners.TrainModelRunner import TrainModelRunner


def train_model():
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--input-data-path", "-i", help="The path to a data file.", type=str, required=True)
    parser.add_argument("--output-model-folder", "-o", help="Prefix to the folder to save output model.", type=str)
    parser.add_argument("--n-epochs", type=int, help="number of epochs of training")
    parser.add_argument("--starting-epoch", type=int, help="the epoch to start training from")
    parser.add_argument("--batch-size", type=int, help="size of the batches")
    parser.add_argument("--lr", type=float, help="adam: learning rate")
    parser.add_argument("--b1", type=float, help="adam: The exponential decay rate for the 1st moment estimates")
    parser.add_argument("--b2", type=float, help="adam: The exponential decay rate for the 2nd moment estimates")
    parser.add_argument("--n-critic", type=int, help="number of training steps for discriminator per generator training step")
    parser.add_argument("--save-interval", type=int, help="interval between saving the model")
    parser.add_argument("--sample-after-training", type=int, help="Number of molecules to sample after training")
    parser.add_argument("--decode-mols-save-path", "-dp", help="Path to save the decoded smiles", type=str)
    parser.add_argument("--message", "-m", type=str, help="The message to print before the training starts")

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    runner = TrainModelRunner(**args)
    runner.run()


if __name__ == "__main__":
    train_model()
