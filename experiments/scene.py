import sys
sys.path.insert(1, '/Users/rushruk/Documents/xai_attempts/CAR/CARs')
import itertools
import logging
import argparse
import torch
import numpy as np
import os
import pandas as pd
from pathlib import Path
from models.scene import SceneClassifierResNet
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.hooks import register_hooks, get_saved_representations, remove_all_hooks
from utils.dataset import load_ade20k_data, generate_scene_concept_dataset, get_ade20k_concept_dataset, get_scene_concepts
from utils.plot import (
    plot_concept_accuracy,
    plot_global_explanation,
    plot_grayscale_saliency,
    plot_attribution_correlation,
    plot_kernel_sensitivity,
    plot_concept_size_impact,
    plot_tcar_inter_concepts,
)
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance, VanillaFeatureImportance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm
from utils.robustness import Attacker

concept_names = []

def train_scene_model(
        batch_size: int = 32,
        model_name: str = "model_scene",
        model_dir: Path = Path.cwd() / f"results/scene/",
        data_dir: Path = Path.cwd() / "data/clip_concepts/"
) -> None:
    logging.info("Fitting Scene Classifier")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    # train_loader, validation_loader = load_scene_data(data_dir=data_dir,
    #                                             batch_size=batch_size)
    train_loader, validation_loader, _ = load_ade20k_data(batch_size=batch_size)
    model = SceneClassifierResNet(name=model_name)
    model.fit(train_loader=train_loader, test_loader=validation_loader, device=device, save_dir=model_dir)


def concept_accuracy(
        random_seeds: list[int],
        # batch_size: int, # TODO: Check if this is needed
        plot: bool = False,
        save_dir: Path = Path.cwd() / "results/scene/concept_accuracy/",
        data_dir: Path = Path.cwd() / "data/gpt_concepts/",
        model_dir: Path = Path.cwd() / f"results/scene/",
        model_name: str = "model"
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seeds[0])
    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    # Load model
    model_dir = model_dir / model_name
    model = SceneClassifierResNet(name=model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    concept_names = get_scene_concepts(data_dir)
    for concept_name, random_seed in itertools.product(concept_names, random_seeds):
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )

        # TODO: Write the generate_scene_concept_dataset function
        X_train, y_train, X_test, y_test = generate_scene_concept_dataset(
            concept_name=concept_name,
            data_dir=data_dir,
            subset_size=200, # Need to check this
            random_seed=random_seed
        )
        # X_train, y_train, _, _ = generate_scene_concept_dataset(
        #     concept_name=concept_name,
        #     data_dir=data_dir,
        #     subset_size=200, # Need to check this
        #     random_seed=random_seed
        # )
        # X_train, y_train = get_ade20k_concept_dataset(
        #     concept_name=concept_name,
        #     subset_size=200,
        #     random_seed=random_seed,
        #     train=True
        # )

        if(X_train is None or len(X_train) < 40):
            remove_all_hooks(handler_train_dic)
            continue

        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)
        module_dic, handler_test_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_test"
        )

        # _, _, X_test, y_test = generate_scene_concept_dataset(
        #     concept_name=concept_name,
        #     data_dir=data_dir,
        #     subset_size=200, # Need to check this
        #     random_seed=random_seed
        # )

        # X_test, y_test = get_ade20k_concept_dataset(
        #     concept_name=concept_name,
        #     subset_size=200,
        #     random_seed=random_seed
        # )

        if(X_test is None):
            remove_all_hooks(handler_test_dic)
            continue

        model(torch.from_numpy(X_test).to(device))
        remove_all_hooks(handler_test_dic)

        for module_name in module_dic:
            logging.info(f"fitting concept classifier for module {module_name}")
            car = CAR(device)
            cav = CAV(device)
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            car.fit(H_train, y_train)
            cav.fit(H_train, y_train)
            hook_name = f"{concept_name}_seed{random_seed}_test_{module_name}"
            H_test = get_saved_representations(hook_name, representation_dir)

            predict_train = car.predict(H_train)
            predict_test = car.predict(H_test)

            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "CAR",
                    accuracy_score(y_train, predict_train),
                    accuracy_score(y_test, predict_test),
                    precision_score(y_test, predict_test, average='macro'),
                    recall_score(y_test, predict_test, average='macro'),
                    f1_score(y_test, predict_test, average='macro')
                ]
            )

            predict_train = cav.predict(H_train)
            predict_test = cav.predict(H_test)

            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "CAV",
                    accuracy_score(y_train, predict_train),
                    accuracy_score(y_test, predict_test),
                    precision_score(y_test, predict_test, average='macro'),
                    recall_score(y_test, predict_test, average='macro'),
                    f1_score(y_test, predict_test, average='macro')
                ]
            )
    results_df = pd.DataFrame(
        results_data,
        columns=["Concept", "Layer", "Seed", "Method", "Train ACC", "Test ACC", "Precision", "Recall", "F1"],
    )
    csv_path = save_dir / "metrics_with_gpt_concepts.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if plot:  # TODO: Check if this is needed
        plot_concept_accuracy(save_dir, None, "scene")
        for concept in concept_names:
            plot_concept_accuracy(save_dir, concept, "scene")

def statistical_significance(
        random_seed: int,
        # batch_size: int, # TODO: Check if this is needed
        # plot: bool = False,
        save_dir: Path = Path.cwd() / "results/scene/statistical_significance/",
        data_dir: Path = Path.cwd() / "data/scene/google_image/google_image/",
        model_dir: Path = Path.cwd() / f"results/scene/",
        model_name: str = "model"
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    # Load model
    model_dir = model_dir / model_name
    model = SceneClassifierResNet(name=model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    concept_names = get_scene_concepts(data_dir)
    for concept_name in concept_names:
        logging.info(f"Working with concept {concept_name}")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )

        # TODO: Write the generate_scene_concept_dataset function
        X_train, y_train, X_test, y_test = generate_scene_concept_dataset(
            concept_name=concept_name,
            data_dir=data_dir,
            subset_size=200, # Need to check this
            random_seed=random_seed
        )
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)


        for module_name in module_dic:
            logging.info(f"Testing concept classifier for module {module_name}")
            car = CAR(device)
            cav = CAV(device)
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAR",
                    car.permutation_test(H_train, y_train),
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAV",
                    cav.permutation_test(H_train, y_train),
                ]
            )
    results_df = pd.DataFrame(
        results_data,
        columns=["Concept", "Layer", "Method", "p-value"],
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        filename="results/scene/scene.log", filemode="w",
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="1")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    # parser.add_argument("--plot", action="store_true")
    # parser.add_argument(
    #     "--concept_sizes", nargs="+", type=int, default=list(range(10, 310, 30))
    # )
    args = parser.parse_args()

    # model_name = f"model_{args.name}"
    model_name = f"model_concept_accuracy" # Need to change this
    concept_accuracy([1,2,3,4,5,6,7,8,9,10], plot=True, model_name=model_name)
    # if args.train:
    #     train_scene_model(args.batch_size, model_name=model_name)
    # if args.name == "concept_accuracy":
    #     concept_accuracy(args.seeds, model_name=model_name)
    # elif args.name == "statistical_significance":
    #     statistical_significance(args.seeds[0], model_name=model_name)
    # else:
    #     raise ValueError(f"Unknown experiment name {args.name}")
