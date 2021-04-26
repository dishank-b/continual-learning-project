import sys
import os
import argparse
from torchvision import transforms
from torch.autograd import Variable
import torch
import numpy as np


def add_args():
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
    parser.add_argument(
        "--nb_tasks", type=int, default=7, help="number of class incremental tasks"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode to reduce model size and number of tasks",
    )
    # TODO add flag to run baselines e.g. ewc and lwf etc...
    # TODO add LR as a paramter in the arguments
    return parser


if __name__ == "__main__":
    parser = add_args()
    args = parser.parse_args()
    print(args)
    mahlanobis_code_path = "src/deep_Mahalanobis_detector"
    if os.path.isdir(mahlanobis_code_path):
        sys.path.append(mahlanobis_code_path)
    else:
        # throw error to clone the repo of deep mahalanobis code
        raise Exception("Deep Mahalanobis code not found!")

    import data_loader
    from src import MahalanobisCompute, create_trainer_model

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

    train_loader, test_loader = data_loader.getTargetDataSet(
        args.dataset, args.batch_size, in_transform, args.dataroot
    )
    if args.reproduce:
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
        dist_compute.compute_all_noise_mahalanobis(
            test_loader, in_transform, args.num_classes
        )
        # now training a logistic regression to detect OOD samples based on its mahalanobis score with reporting its performance
        dist_compute.cross_validate()

        # TODO add joint training
    elif args.continual:
        from src import create_model, OODSequoia
        from sequoia.settings.passive.cl import DomainIncrementalSetting

        from sequoia.common import Config

        if args.debug:
            model = create_model("debug", args.num_classes)
            dataset = "fashionmnist"
            nb_tasks = 5
            epochs = 3
            dist_compute = None
        else:
            model = create_model(args.net_type, args.num_classes)
            dataset = args.dataset
            nb_tasks = args.nb_tasks
            epochs = args.epochs
            dist_compute = MahalanobisCompute(args, model)
        # TODO add lwf and ewc arguments for the bseline
        hparams = OODSequoia.HParams(
            start_lr=3e-4,
            epochs=epochs,
            batch_size=args.batch_size,
            ood_regularizer=0.75,
            lwf_regularizer=0,
            temperature_lwf=2,
            wandb_logging=False,
        )
        method = OODSequoia(
            test_loader.dataset, model, dist_compute, in_transform, hparams
        )
        setting = DomainIncrementalSetting(
            dataset=dataset,
            nb_tasks=nb_tasks,
            batch_size=args.batch_size,
        )
        results = setting.apply(method, config=Config(data_dir="data"))
        # now add plots coming from results
        plots_dict = results.make_plots()
        plots_dict["task_metrics"].savefig(os.path.join(args.outf, "results_plot.jpg"))
        summary = results.summary()
        print(summary)
        with open(os.path.join(args.outf, "results.txt"), "w") as f:
            f.write(summary)
        # TODO add WANDB support

    else:
        raise NotImplementedError(
            "Add continual or reproduce flag !! No default scenario"
        )

