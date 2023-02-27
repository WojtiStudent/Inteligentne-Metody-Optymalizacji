import pandas as pd
import scipy.spatial

def load_data(path, lines_to_skip):
    with open(path, 'r') as f:
        for _ in range(lines_to_skip):
            f.readline()
        df = pd.read_csv(f, sep=' ', header=None, names=['x', 'y'])
        df = df.iloc[:-1, :]
    return df

def create_graph(df):
    distance_matrix = scipy.spatial.distance.cdist(df, df, 'euclidean')
    dfm = pd.DataFrame(distance_matrix).round().astype(int)
    return dfm

if __name__ == '__main__':
    # Test
    df = load_data('data\kroa100.tsp', 6)
    print(df)
    distance_matrix = create_graph(df)
    print(distance_matrix)