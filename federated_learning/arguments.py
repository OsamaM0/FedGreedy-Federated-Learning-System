from federated_learning.model.nets import FashionMNISTCNN, HAPTDNN, Cifar10CNN
import torch
import json

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)

class Arguments:

    def __init__(self, logger):
        self.logger = logger
        self.algorithm = "fed_avg"
        self.batch_size = 6
        self.test_batch_size = 1000
        self.cr = 5
        self.epoch = 200
        self.lr = 0.01
        self.mu = 0.01
        self.momentum = 0.9
        self.cuda = True
        self.shuffle = False
        self.log_interval = 100
        self.kwargs = {}

        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.5
        self.min_lr = 1e-10


        self.round_worker_selection_strategy = None
        self.round_worker_selection_strategy_kwargs = None

        self.save_model = True
        self.save_cr_interval = 1
        self.save_model_path = "models"
        self.save_stats_path = "stats"
        self.cr_save_start_suffix = "start"
        self.cr_save_end_suffix = "end"

        self.num_workers = 50
        self.num_poisoned_workers = 0

        self.struggling_workers_percentage = 0
        self.struggling_epochs_percentage = 0

        # self.net = Cifar10CNN
        # self.net = FashionMNISTCNN
        self.net = HAPTDNN

        self.train_data_loader_pickle_path = "data_loaders/fashion-mnist/train_data_loader.pickle"
        self.test_data_loader_pickle_path = "data_loaders/fashion-mnist/test_data_loader.pickle"

        # self.train_data_loader_pickle_path = "data_loaders/mnist/train_data_loader.pickle"
        # self.test_data_loader_pickle_path = "data_loaders/mnist/test_data_loader.pickle"
        # self.train_data_loader_pickle_path = "data_loaders/cifar10/train_data_loader.pickle"
        # self.test_data_loader_pickle_path = "data_loaders/cifar10/test_data_loader.pickle"
        # self.train_data_loader_pickle_path = "data_loaders/hapt/train_data_loader.pickle"
        # self.test_data_loader_pickle_path = "data_loaders/hapt/test_data_loader.pickle"
        # self.data_distribution = "non_iid"
        self.data_distribution = "iid"
        self.loss_function = torch.nn.CrossEntropyLoss

        self.default_model_folder_path = "default_models"

        self.data_path = "data"

    def set_algorithm(self, algorithm):
        """Parameters: algorithm: [str] = ["fed_avg", "fed_prox", "fed_max"]"""
        self.algorithm = algorithm

    def get_algorithm(self):
        return self.algorithm

    def set_epoch(self, epoch):
        self.epoch = epoch
    def get_epoch(self):
        return self.epoch

    def set_struggling_workers_percentage(self, struggling_workers_percentage):
        self.struggling_workers_percentage = struggling_workers_percentage
    def get_struggling_workers_percentage(self):
        return self.struggling_workers_percentage
    def get_struggling_epochs_percentage(self):
        return self.struggling_epochs_percentage
    def set_struggling_epochs_percentage(self, struggling_epochs_percentage):
        self.struggling_epochs_percentage = struggling_epochs_percentage

    def get_round_worker_selection_strategy(self):
        return self.round_worker_selection_strategy

    def get_round_worker_selection_strategy_kwargs(self):
        return self.round_worker_selection_strategy_kwargs

    def set_round_worker_selection_strategy_kwargs(self, kwargs):
        self.round_worker_selection_strategy_kwargs = kwargs

    def set_client_selection_strategy(self, strategy):
        self.round_worker_selection_strategy = strategy

    def get_data_path(self):
        return self.data_path

    def get_cr_save_start_suffix(self):
        return self.cr_save_start_suffix

    def get_cr_save_end_suffix(self):
        return self.cr_save_end_suffix

    def set_train_data_loader_pickle_path(self, path):
        self.train_data_loader_pickle_path = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path

    def set_test_data_loader_pickle_path(self, path):
        self.test_data_loader_pickle_path = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path

    def get_cuda(self):
        return self.cuda

    def get_scheduler_step_size(self):
        return self.scheduler_step_size

    def get_scheduler_gamma(self):
        return self.scheduler_gamma

    def get_min_lr(self):
        return self.min_lr

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_num_cr(self):
        return self.cr

    def set_num_poisoned_workers(self, num_poisoned_workers):
        self.num_poisoned_workers = num_poisoned_workers

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_model_save_path(self, save_model_path):
        self.save_model_path = save_model_path

    def set_stats_save_path(self, save_stats_path):
        self.save_stats_path = save_stats_path


    def get_logger(self):
        return self.logger

    def get_loss_function(self):
        return self.loss_function

    def get_net(self):
        return self.net

    def get_num_workers(self):
        return self.num_workers

    def get_num_poisoned_workers(self):
        return self.num_poisoned_workers

    def get_learning_rate(self):
        return self.lr

    def get_momentum(self):
        return self.momentum
    def get_mu(self):
        return self.mu

    def get_shuffle(self):
        return self.shuffle

    def get_batch_size(self):
        return self.batch_size

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_log_interval(self):
        return self.log_interval

    def get_save_model_folder_path(self):
        return self.save_model_path

    def get_save_stats_folder_path(self):
        return self.save_stats_path

    def get_data_distribution(self):
        return self.data_distribution

    def set_data_distribution(self, data_distribution):
        self.data_distribution = data_distribution


    def get_learning_rate_from_cr(self, round_idx):
        lr = self.lr * (self.scheduler_gamma ** int(round_idx / self.scheduler_step_size))

        if lr < self.min_lr:
            self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

            return self.min_lr

        self.logger.debug("LR: {}".format(lr))

        return lr

    def should_save_model(self, round_idx):
        """
        Returns true/false models should be saved.

        :param round_idx: current training round index
        :type round_idx: int
        """
        if not self.save_model:
            return False

        if round_idx == 1 or round_idx % self.save_cr_interval == 0:
            return True

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Test Batch Size: {}\n".format(self.test_batch_size) + \
               "Communication Round: {}\n".format(self.cr) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Momentum: {}\n".format(self.momentum) + \
               "CUDA Enabled: {}\n".format(self.cuda) + \
               "Shuffle Enabled: {}\n".format(self.shuffle) + \
               "Log Interval: {}\n".format(self.log_interval) + \
               "Scheduler Step Size: {}\n".format(self.scheduler_step_size) + \
               "Scheduler Gamma: {}\n".format(self.scheduler_gamma) + \
               "Scheduler Minimum Learning Rate: {}\n".format(self.min_lr) + \
               "Client Selection Strategy: {}\n".format(self.round_worker_selection_strategy) + \
               "Model Saving Enabled: {}\n".format(self.save_model) + \
               "Model Saving Interval: {}\n".format(self.save_cr_interval) + \
               "Model Saving Path (Relative): {}\n".format(self.save_model_path) + \
               "CR Save Start Prefix: {}\n".format(self.cr_save_start_suffix) + \
               "CR Save End Suffix: {}\n".format(self.cr_save_end_suffix) + \
               "Number of Clients: {}\n".format(self.num_workers) + \
               "Number of Poisoned Clients: {}\n".format(self.num_poisoned_workers) + \
               "NN: {}\n".format(self.net) + \
               "Train Data Loader Path: {}\n".format(self.train_data_loader_pickle_path) + \
               "Test Data Loader Path: {}\n".format(self.test_data_loader_pickle_path) + \
               "Loss Function: {}\n".format(self.loss_function) + \
               "Default Model Folder Path: {}\n".format(self.default_model_folder_path) + \
               "Data Path: {}\n".format(self.data_path)
