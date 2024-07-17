import pandas as pd
import time

def store_replay(replays):
    """
        Get the replays from all iterations. 
        Save each replay seperate to a csv file.

    Args:
        replays (list): List with replays.
    """
    # TODO: maybe add other stats to replay
    indices = [i for i, replay in enumerate(replays[0]) if replay == "reset"]
    for i in range(0, len(indices) + 1):
        # TODO: clean this
        if i == 0:
            a = 0
        else:
            a = indices[i - 1]
        
        if i == len(indices):
            b = len(replays[0])
        else:
            b = indices[i]

        new_run = replays[0][a:b]
        if "reset" in new_run:
            new_run.remove("reset")
        
        coordinates = []
        moves = []

        for step in new_run:
            # Can put "metadata" in line one.
            coordinates.append(step[0])
            moves.append(step[1])

        df = pd.DataFrame({"coordinates": coordinates, "moves": moves})
        # Add param in title soon
        filename = "replays/" + "rpl" + str(i) + "-" + str(time.strftime("%Y%m%d-%H%M%S")) + ".csv"
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
    return df["coordinates"].to_list(), df["moves"].to_list()