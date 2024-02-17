from copy import deepcopy

from client_udpate_strategy import  ClientUpdateStrategy
import torch
class AvgClientUpdate(ClientUpdateStrategy):
    def update_client(self,  client, server_model, history, learning_rate, epochs, loss_func, mu=None, kwargs=None):
        local_model = deepcopy(server_model)
        local_loss = 0
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)

        for i in range(epochs):
            epoch_loss = 0

            # Training
            for batch in client:
                pass
                # loss = train(local_model, batch, loss_func)
                # # if (type == "prox"):
                # #     loss += (mu / 2) * (diff_squared_sum(local_model, server_model))
                # # # if(type=="q-ffl"):
                # # #     loss = (1/(q+1))*((loss)**(q+1))
                #
                # epoch_loss += loss
                #
                # loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()

            epoch_loss = epoch_loss / len(client)
            local_loss = epoch_loss

        history.log_client(local_model, client, local_loss)
        return history