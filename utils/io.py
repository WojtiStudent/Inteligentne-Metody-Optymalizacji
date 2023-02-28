import pandas as pd
import scipy.spatial
import numpy as np

def load_data(path: str, lines_to_skip: int) -> pd.DataFrame:
    """
    Loads problem data from a file and returns a pandas DataFrame.
    :param path: path to the file
    :param lines_to_skip: number of lines to skip
    :return: pandas DataFrame
    """
    with open(path, 'r') as f:
        for _ in range(lines_to_skip):
            f.readline()
        df = pd.read_csv(f, sep=' ', header=None, names=['x', 'y'])
        # Remove last row (EOF)
        df = df.iloc[:-1, :]
    return df

def create_graph(df: pd.DataFrame) -> np.ndarray:
    """
    Creates a distance graph from a pandas DataFrame with coordinates of vertexes.
    :param df: pandas DataFrame with coordinates of vertexes
    :return: numpy array with distancess
    """
    distance_matrix = scipy.spatial.distance.cdist(df, df, 'euclidean')
    # Set diagonal to infinity to avoid self-loops
    dfm = pd.DataFrame(distance_matrix).round().astype(int)
    np.fill_diagonal(distance_matrix, np.inf)
    return dfm.values
