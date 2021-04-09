import sys
import os
import argparse
from torchvision import transforms
from torch.autograd import Variable
import torch
import numpy as np


if __name__ == "__main__":
    # TODO add flag for continual learning
    parser = argparse.ArgumentParser(description="PyTorch code: Mahalanobis detector")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        metavar="N",
        help="batch size for data loader",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, metavar="N", help="training epochs"
    )
    parser.add_argument("--dataset", required=True, help="cifar10 | cifar100 | svhn")
    parser.add_argument("--dataroot", default="./data", help="path to dataset")
    parser.add_argument("--outf", default="./output/", help="folder to output results")
    parser.add_argument("--num_classes", type=int, default=10, help="the # of classes")
    parser.add_argument("--net_type", required=True, help="resnet | densenet")
    parser.add_argument(
        "--reproduce", action="store_true", help="Reproducing Deep Mahalanobis compute"
    )
    parser.add_argument(
        "--continual",
        action="store_true",
        help="Class incremental scenario with deep Mahalanobis  ",
    )
    # TODO add a flag to cross-validate or to simply do the training and another flag for CL
    args = parser.parse_args()
    print(args)
    mahlanobis_code_path = "src/deep_Mahalanobis_detector"
    if os.path.isdir(mahlanobis_code_path):
        sys.path.append(mahlanobis_code_path)
    else:
        # throw error to clone the repo of deep mahalanobis code
        raise Exception("Deep Mahalanobis code not found!")

    import data_loader
    from src import MahalanobisCompute, create_trainer_model, ContinualTrainer

    network_f_name = args.net_type + "_" + args.dataset + ".pth"
    pre_trained_net = "./pre_trained/" + network_f_name
    args.outf = args.outf + args.net_type + "_" + args.dataset + "/"
    if os.path.isdir(args.outf) == False:
        os.makedirs(args.outf)

    if args.net_type == "densenet":
        in_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (125.3 / 255, 123.0 / 255, 113.9 / 255),
                    (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0),
                ),
            ]
        )
    elif args.net_type == "resnet":
        in_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        raise Exception("Network type {} doesnt exist !!".format(args.net_type))
    # load dataset
    print("load target data: ", args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.reproduce:
        train_loader, test_loader = data_loader.getTargetDataSet(
            args.dataset, args.batch_size, in_transform, args.dataroot
        )
        # get model and trainer to train the network using pytorch lightning
        trainer, network = create_trainer_model(
            args.net_type,
            train_loader,
            test_loader,
            args.num_classes,
            args.epochs,
            network_f_name,
            batch_size=args.batch_size,
        )
        # training
        trainer.fit(network)

        # creating Mahalanobis compute object
        dist_compute = MahalanobisCompute(args, network.get_base_model())

        # computing mean and precision
        dist_compute.compute_data_stats(train_loader, args.num_classes)
        # computing and saving mahalanobis scores
        dist_compute.compute_all_noise_mahalanobis(test_loader, in_transform, args.num_classes)
        # now training a logistic regression to detect OOD samples based on its mahalanobis score with reporting its performance
        dist_compute.cross_validate()

        # TODO add joint training
    elif args.continual:
        from torch.utils.data import DataLoader
        from continuum import ClassIncremental
        from continuum.datasets import CIFAR10, CIFAR100
        from continuum.tasks import split_train_val
        from src import create_model

        # class incremental scenario
        if args.dataset == "cifar10":
            train_dataset = CIFAR10(args.dataroot, train=True, download=True)
            test_dataset = CIFAR10(args.dataroot, train=False, download=True)
        elif args.dataset == "cifar100":
            train_dataset = CIFAR100(args.dataroot, train=True, download=True)
            test_dataset = CIFAR100(args.dataroot, train=False, download=True)
        elif args.dataset == "svhn":
            raise NotImplementedError(
                "SVHN dataset not supported in Continuum and continual learning scenario"
            )
        # TODO maybe we can change the scenario depending on the results or the dataset or add a flag for increments
        scenario = ClassIncremental(train_dataset, increment=1, initial_increment=3)
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=args.batch_size
        )
        model = create_model(args.net_type, args.num_classes)
        dist_compute = MahalanobisCompute(args, model)
        trainer = ContinualTrainer(
            scenario,
            test_loader,
            model,
            0.1,
            args.batch_size,
            dist_compute,
            in_transform,
            args.epochs
        )
        trainer.fit()

    else:
        raise NotImplementedError(
            "Add continual or reproduce flag !! No default scenario"
        )

