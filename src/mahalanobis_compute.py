import torch
from torch.autograd import Variable
import numpy as np
import lib_generation
import data_loader as DataLoader
import os
import lib_regression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


class MahalanobisCompute:
    """Wrapper on top of deep_mahalanobis distance to continually update covariance and mean
    """    
    def __init__(self, args, base_model):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = base_model.to(self.device)
        self.feature_list = None
        self.out_dist_list = None
        self.num_output = None
        self.sample_mean = None
        self.precision = None
        self.ood_classifiers = None
        self._init_information()

    def _init_information(self):
        """Initialization of information
        """        
        if self.args.dataset == "svhn":
            self.out_dist_list = ["cifar10", "imagenet_resize", "lsun_resize"]
        else:
            self.out_dist_list = ["svhn", "imagenet_resize", "lsun_resize"]
        self.model.eval()
        temp_x = torch.rand(2, 3, 32, 32).to(self.device)
        temp_x = Variable(temp_x)
        temp_list = self.model.feature_list(temp_x)[1]
        self.num_output = len(temp_list)
        self.feature_list = np.empty(self.num_output)
        count = 0
        for out in temp_list:
            self.feature_list[count] = out.size(1)
            count += 1
        self.ood_classifiers = {}

    def update_network(self, model):
        """updates network used to compute mean and covariance

        Args:
            model (nn.Module): new task's model
        """        
        self.model = model.to(self.device)
        self._init_information()

    def compute_data_stats(self, train_loader, num_classes):
        """Computes mean and covariance

        Args:
            train_loader (torch.dataloader): training data loader
            num_classes (int): number of classes
        """        
        print("get sample mean and covariance")
        self.sample_mean, self.precision = self.sample_estimator(
            train_loader, num_classes
        )

    def sample_estimator(self, train_loader, num_classes):
        sample_mean, precision = lib_generation.sample_estimator(
            self.model, num_classes, self.feature_list, train_loader
        )
        return sample_mean, precision

    def compute_all_noise_mahalanobis(
        self,
        data_loader,
        in_transform,
        num_classes,
        m_list=[0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005],
    ):
        assert self.sample_mean is not None
        print("get Mahalanobis scores")
        for magnitude in m_list:
            print("Noise: " + str(magnitude))
            self.compute_mahalanobis(data_loader, in_transform, magnitude, num_classes)

    def compute_mahalanobis(self, data_loader, in_transform, magnitude, num_classes):
        for i in range(self.num_output):
            M_in = lib_generation.get_Mahalanobis_score(
                self.model,
                data_loader,
                num_classes,
                self.args.outf,
                True,
                self.args.net_type,
                self.sample_mean,
                self.precision,
                i,
                magnitude,
            )
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate(
                    (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1
                )

        for out_dist in self.out_dist_list:
            out_test_loader = DataLoader.getNonTargetDataSet(
                out_dist, self.args.batch_size, in_transform, self.args.dataroot
            )
            print("Out-distribution: " + out_dist)
            for i in range(self.num_output):
                M_out = lib_generation.get_Mahalanobis_score(
                    self.model,
                    out_test_loader,
                    num_classes,
                    self.args.outf,
                    False,
                    self.args.net_type,
                    self.sample_mean,
                    self.precision,
                    i,
                    magnitude,
                )
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate(
                        (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1,
                    )

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            (
                Mahalanobis_data,
                Mahalanobis_labels,
            ) = lib_generation.merge_and_generate_labels(
                Mahalanobis_out, Mahalanobis_in
            )
            file_name = os.path.join(
                self.args.outf,
                "Mahalanobis_%s_%s_%s.npy"
                % (str(magnitude), self.args.dataset, out_dist),
            )
            Mahalanobis_data = np.concatenate(
                (Mahalanobis_data, Mahalanobis_labels), axis=1
            )
            np.save(file_name, Mahalanobis_data)

    def cross_validate(self, m_list=[0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005], save_classifier=False):
        list_best_results_out, list_best_results_index_out = [], []
        score_list = ["Mahalanobis_{}".format(str(margin)) for margin in m_list]

        for out in self.out_dist_list:
            print("Out-of-distribution: ", out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(
                    score, self.args.dataset, out, self.args.outf
                )
                X_train, X_test, Y_train, Y_test = train_test_split(
                    total_X, total_Y, train_size=1000, stratify=total_Y
                )
                lr, results = self.fit_regression(X_train, Y_train)
                if best_tnr < results["TMP"]["TNR"]:
                    best_tnr = results["TMP"]["TNR"]
                    best_index = score
                    best_result = lib_regression.detection_performance(
                        lr, X_test, Y_test, self.args.outf
                    )
            if best_result != 0:
                list_best_results_out.append(best_result)
                list_best_results_index_out.append(best_index)
            if save_classifier:
                ood_clf, _ = self.fit_regression(total_X, total_Y, validate=False)
                self.ood_classifiers[out] = ood_clf
        count_out = 0
        for results in list_best_results_out:
            print("out_distribution: " + self.out_dist_list[count_out])
            print("\n{val:6.2f}".format(val=100.0 * results["TMP"]["TNR"]), end="")
            print(" {val:6.2f}".format(val=100.0 * results["TMP"]["AUROC"]), end="")
            print(" {val:6.2f}".format(val=100.0 * results["TMP"]["DTACC"]), end="")
            print(" {val:6.2f}".format(val=100.0 * results["TMP"]["AUIN"]), end="")
            print(" {val:6.2f}\n".format(val=100.0 * results["TMP"]["AUOUT"]), end="")
            print("Input noise: " + list_best_results_index_out[count_out])
            print("")
            count_out +=1
        return list_best_results_out

    def fit_regression(self, x_train, y_train, validate=True):
        if validate: 
            # rectified split of train and validation to avoid imbalance
            X_train, X_val_for_test, Y_train, Y_val_for_test = train_test_split(
                x_train, y_train, test_size=0.5, stratify=y_train
            )
        else:
            X_train = x_train
            Y_train = y_train
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
        results = None
        if validate:
            results = lib_regression.detection_performance(
                lr, X_val_for_test, Y_val_for_test, self.args.outf
            )
        return lr, results
    
    def compute_task_risk(self, data_loader, magnitude, num_classes):
        # compute mahalanobis score
        for i in range(self.num_output):
            data_mahalanobis = lib_generation.get_Mahalanobis_score(
                self.model,
                data_loader,
                num_classes,
                self.args.outf,
                True,
                self.args.net_type,
                self.sample_mean,
                self.precision,
                i,
                magnitude,
            )
            data_mahalanobis = np.asarray(data_mahalanobis, dtype=np.float32)
            if i == 0:
                task_mahalanobis = data_mahalanobis.reshape((data_mahalanobis.shape[0], -1))
            else:
                task_mahalanobis = np.concatenate(
                    (task_mahalanobis, data_mahalanobis.reshape((data_mahalanobis.shape[0], -1))), axis=1
                )
        # now computing scvore based on average of all ood clfs
        predicted_score = 0
        for clf_name in self.ood_classifiers:
            predicted_score += np.mean(self.ood_classifiers[clf_name].predict_proba(task_mahalanobis))
        predicted_score /= len(self.ood_classifiers)
        return predicted_score
