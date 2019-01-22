import seaborn as sns
import pandas as pd


def _nameFromClustering(clustering):
    return clustering.method_name+"_"+str(clustering.method_parameters)

def plot_sim_matrix(clusterings,scoring):
    """

    :param clusterings: list of clusterings to compare
    :param scoring: the scoring function to use
    """
    forDF= []
    for c in clusterings:
        cID = _nameFromClustering(c)
        for c2 in clusterings:
            c2ID = _nameFromClustering(c2)
            forDF.append([cID,c2ID,scoring(c,c2)])
    df = pd.DataFrame(columns=["com1","com2","score"],data=forDF)
    df = df.pivot("com1", "com2", "score")
    return sns.clustermap(df)