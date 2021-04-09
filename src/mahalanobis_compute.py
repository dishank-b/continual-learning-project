import torch
from torch.autograd import Variable
import numpy as np
import lib_generation
import data_loader
import os
import lib_regression
from sklearn.linear_model import LogisticRegressionCV


class MahalanobisCompute:
    def __init__(self, args, base_model):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = base_model.to(self.device)
        self.feature_list = None
        self.out_dist_list = None
        self.num_output = None
        self.sample_mean = None
        self.precision = None
        self._init_information()

    def _init_information(self):
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
    
    def update_network(self, model):
        self.model = model.to(self.device)
        self._init_information()

    def compute_data_stats(self, train_loader, num_classes):
        print("get sample mean and covariance")
        self.sample_mean, self.precision = self.sample_estimator(train_loader, num_classes)

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
            self.compute_mahalanobis(data_loader, in_transform, magnitude,num_classes, True)

    def compute_mahalanobis(self, data_loader, in_transform, magnitude, num_classes, save=False):
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
            out_test_loader = data_loader.getNonTargetDataSet(
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
            if save:
                np.save(file_name, Mahalanobis_data)
            return Mahalanobis_data[:, 0], Mahalanobis_data[:, 1]

    def cross_validate(self, m_list=[0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]):
        list_best_results_out, list_best_results_index_out = [], []
        score_list = ["Mahalanobis_{}".format(str(margin)) for margin in m_list]

        for out in self.out_dist_list:
            print("Out-of-distribution: ", out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(
                    score, self.args.dataset, out, self.args.outf
                )

                X_train, Y_train, X_test, Y_test = lib_regression.block_split(
                    total_X, total_Y, out
                )
                lr, results, X_test, Y_test = self.fit_regression(X_train, Y_train)
                if best_tnr < results["TMP"]["TNR"]:
                    best_tnr = results["TMP"]["TNR"]
                    best_index = score
                    best_result = lib_regression.detection_performance(
                        lr, X_test, Y_test, self.args.outf
                    )
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        count_out = 0
        for results in list_best_results_out:
            print("out_distribution: " + self.out_dist_list[count_out])
            print("\n{val:6.2f}".format(val=100.0 * results["TMP"]["TNR"]), end="")
            print(" {val:6.2f}".format(val=100.0 * results["TMP"]["AUROC"]), end="")
            print(" {val:6.2f}".format(val=100.0 * results["TMP"]["DTACC"]), end="")
            print(" {val:6.2f}".format(val=100.0 * results["TMP"]["AUIN"]), end="")
            print(" {val:6.2f}\n".format(val=100.0 * results["TMP"]["AUOUT"]), end="")
            print("Input noise: " + list_best_results_out[count_out])
            print("")

    def fit_regression(self, x_train, y_train):
        X_train = np.concatenate((x_train[:500], x_train[1000:1500]))
        Y_train = np.concatenate((y_train[:500], y_train[1000:1500]))
        X_val_for_test = np.concatenate((x_train[500:1000], x_train[1500:]))
        Y_val_for_test = np.concatenate((y_train[500:1000], y_train[1500:]))
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
        results = lib_regression.detection_performance(
            lr, X_val_for_test, Y_val_for_test, self.args.outf
        )
        return lr, results

