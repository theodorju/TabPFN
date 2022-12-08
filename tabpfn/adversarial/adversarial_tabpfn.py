"""Minimal code to perform adversarial attack on TabPFN model trained on breast cancer dataset."""
import argparse
from tabpfn.adversarial.adversarial import *
from sklearn.datasets import *


def adversarial_sklearn_datasets(dataset_fn=None, lr=0.001, num_steps=24):
    adv = AdversarialTabPFNInterface(dataset_fn=dataset_fn, num_steps=num_steps, lr=lr, print_every=4, save_results=True)
    _, _, _, _, _ = adv.adversarial_attack()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate for the adversarial attack.",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "-s",
        "--steps",
        help="Number of adversarial attack steps.",
        type=int,
        default=20
    )

    args = parser.parse_args()
    lr, num_steps = args.learning_rate, args.steps

    ds = load_breast_cancer
    adversarial_sklearn_datasets(ds, lr, num_steps)
