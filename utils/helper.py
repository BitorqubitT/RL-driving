import pandas as pd
import time
import matplotlib.pyplot as plt
import torch
from IPython import display

def store_replay(replays):
    """
        Get the replays from all iterations. 
        Save each replay seperate to a csv file.

    Args:
        replays (list): List with replays.
    """

    df = pd.DataFrame(replays, columns = ["x", "y", "heading", "speed", "rewards", "moves"])
    # Add param in title soon
    filename = "replays/" + "rpl" + "-" + str(time.strftime("%Y%m%d-%H%M%S")) + ".csv"
    df.to_csv(filename)

def read_replay(file_name):
    """
        Get list of replays, load these.

    Args:
        file_name (str): Path to file containing replay.

    Returns:
        Lists: Two lists containing coordinates and moves.
    """
    df = pd.read_csv(file_name)
    return df

def plot_durations(episode_durations):
    """
        Shows a plot during training.

    Args:
        episode_durations (_type_): _description_
    """
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    display.display(plt.gcf())
    display.clear_output(wait=True)



