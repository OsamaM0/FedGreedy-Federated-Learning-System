from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import math

def plot_data_distribution(train_data_loader):
    num_clients = len(train_data_loader)
    dataset_classes = set(range(10))  # Get the unique classes in the dataset
    print(dataset_classes)
    grid_size = (math.ceil(math.sqrt(num_clients)))  # Calculate the size of the grid and round up

    fig, axs = plt.subplots(grid_size-1, grid_size, figsize=(15, 15))  # Adjust figsize for more square subplots

    colors = plt.cm.viridis(np.linspace(0, 1, num_clients))  # Generate different colors

    for idx, client_data in enumerate(train_data_loader):
        # Separate data points and labels, and flatten each label
        labels = client_data[1].tolist()
        # Count occurrences of each number
        counter = Counter(labels)

        # Extract unique numbers and their occurrences
        numbers = list(counter.keys())
        occurrences = list(counter.values())

        # Plot the distribution of labels
        axs[idx // grid_size, idx % grid_size].bar(numbers, occurrences, color=colors[idx])
        axs[idx // grid_size, idx % grid_size].set_title(f'Client {idx}', fontsize=10)  # Set smaller font size
        axs[idx // grid_size, idx % grid_size].set_xticks(list(dataset_classes))
        # axs[idx // grid_size, idx % grid_size].set_xlabel('Classes', fontsize=8)  # Set smaller font size
        axs[idx // grid_size, idx % grid_size].set_ylabel('Num of Samples', fontsize=8)  # Set smaller font size
        axs[idx // grid_size, idx % grid_size].set_ylim(0, 1000)  # Set y-axis limits from 0 to 1000
        axs[idx // grid_size, idx % grid_size].set_yticks(np.arange(0, 1000, 200))  # Set y-axis ticks from 0 to 1000 with step size 100

    plt.tight_layout(pad=3.0)  # Increase space between subplots

    plt.show()

