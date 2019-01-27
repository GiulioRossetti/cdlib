import seaborn as sns
import pandas as pd
import nclib

def plot_sim_matrix(clusterings,scoring):
    """
    Plot a similarity matrix between a list of clusterings, using the provided scoring function.

    :param clusterings: list of clusterings to compare
    :param scoring: the scoring function to use
    """
    forDF= []
    for c in clusterings:
        cID = c.get_description()
        for c2 in clusterings:
            c2ID = c2.get_description()
            forDF.append([cID,c2ID,scoring(c,c2)])
    df = pd.DataFrame(columns=["com1","com2","score"],data=forDF)
    df = df.pivot("com1", "com2", "score")
    return sns.clustermap(df)

def plot_com_stat(com_clusters,com_fitness):
    """
    Plot the distribution of a property among all communities for a clustering, or a list of clusterings (violin-plots)

    :param com_clusters: list of clusterings to compare, or a single clustering
    :param com_fitness: the fitness/community property to use
    """
    if isinstance(com_clusters, nclib.classes.clustering.Clustering):
        com_clusters = [com_clusters]

    allVals = []
    allNames = []
    for c in com_clusters:
        prop = com_fitness(c.graph,c,summary=False)
        allVals+= prop
        allNames+= [c.get_description()]*len(prop)

    ax = sns.violinplot(allNames,allVals)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    return ax

def plot_com_properties_relation(com_clusters,com_fitness_x,com_fitness_y, log_x = False, log_y = False, **kwargs):
    """
    Plot the relation between two properties/fitness function of a clustering

    :param com_clusters:  clustering(s) to analyze (cluster or cluster list)
    :param com_fitness_x: first fitness/community property
    :param com_fitness_y: first fitness/community property
    :param log_x: if True, plot the x axis in log scale
    :param log_y: if True, plot the y axis in log scale
    :param kwargs: parameters for the seaborn lmplot
    """
    if isinstance(com_clusters,nclib.classes.clustering.Clustering):
        com_clusters = [com_clusters]

    for_df = []

    for c in com_clusters:
        x = com_fitness_x(c.graph, c, summary=False)
        y = com_fitness_y(c.graph, c, summary=False)
        for i, vx in enumerate(x):
            for_df.append([c.get_description(), vx, y[i]])

    df = pd.DataFrame(columns=["Method", "Property1", "Property2"], data=for_df)
    ax = sns.lmplot(x="Property1", y="Property2", data=df, hue="Method", fit_reg=False, x_bins=100, **kwargs)
    if (log_x):
        ax.set_xscale("log")
    if (log_y):
        ax.set_yscale("log")
    return ax

def plot_scoring(graphs,ref_partitions,graph_names,methods,scoring=nclib.evaluation.adjusted_mutual_information,nbRuns=5):
    """
    Plot the scores obtained by a list of methods on a list of graphs.

    :param graphs: list of graphs on which to make computations
    :param ref_partitions: list of reference clusterings corresponding to graphs
    :param graph_names: list of the names of the graphs to display
    :param methods: list of functions that take a graph as input and return a Clustering as output
    :param scoring: the scoring function to use, default anmi
    :param nbRuns: number of runs to do for each method on each graph
    """
    forDF = []
    for i,g in enumerate(graphs):
        for m in methods:
            for r in range(nbRuns):
                partition = m(g)
                score = scoring(partition,ref_partitions[i])
                forDF.append([graph_names[i],score,partition.get_description()])
    df = pd.DataFrame(columns=["graph","score","method"],data=forDF)
    ax = sns.lineplot(x="graph",y="score",hue="method",data=df,legend="brief")
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    return ax