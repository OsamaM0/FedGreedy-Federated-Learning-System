from abc import abstractmethod

class ClientUpdateStrategy:

    @abstractmethod
    def update_client(self,  client, server_model, history, learning_rate, epochs, loss_func, mu=None, kwargs=None):
        """
        :param client: Client
        :type client: Client
        :param server_model: Server model
        :type server_model: torch.nn
        :param history: History of server models
        :type history: list(torch.nn)
        :param learning_rate: Learning rate
        :type learning_rate: float
        :param epochs: Number of epochs
        :type epochs: int
        :param mu: Mu
        :type mu: float
        :param loss_func: Loss function
        :type loss_func: torch.nn
        :param kwargs: dict
        """
        raise NotImplementedError("update_client() not implemented")
