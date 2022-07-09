import logging
import typing as tp

from ..dataset import DialogueDataset, Utterance, Dialogue
from ..embedders import SentenceEmbedder
from ..clustering import SubClustering, KMeansClustering, OneViewClustering
from ..clustering.filters import speaker_filter
from ..dialogue_graph import FrequencyDialogueGraph
from ..cluster2vec import full_merge_build


def clusterize(
        train: DialogueDataset,
        embedder: SentenceEmbedder,
        n_clusters: int = 60
) -> SubClustering:
    """Embed dialogue, create clusters out of dialogue utterances.

    Args:
        train:
        embedder:
        n_clusters: Number of clusters to split into

    Returns: clustering
    """
    logging.info("Encoding...")
    train_embeddings = embedder.encode_dataset(train)
    logging.info(f"Encoding done. Clustering with parameters: n_clusters={n_clusters}")

    clustering = SubClustering(train, KMeansClustering, speaker_filter,
                               {'n_clusters': n_clusters // 2}).fit(train_embeddings)
    logging.info(f'Clustering done. Total {clustering.get_nclusters()} clusters.')

    return clustering


def dgac_one_stage(
        train: DialogueDataset,
        embedder: SentenceEmbedder,
        clustering: tp.Optional[OneViewClustering] = None,
        n_clusters: int = 60
) -> FrequencyDialogueGraph:
    """Build dialogue graph in one stage.

    Args:
        train:
        embedder:
        clustering: pre-computed clustering of n_clusters size
        n_clusters:

    Returns: dialogue graph

    """
    if clustering is not None:
        assert clustering.get_nclusters() == n_clusters
    logging.info(f"Building graph with parameters: n_clusters={n_clusters}")
    graph = FrequencyDialogueGraph(train, embedder, clustering or clusterize(train, embedder, n_clusters))
    graph.build()
    logging.info("Graph built.")
    return graph


def dgac_two_stage(
        train: DialogueDataset,
        embedder: SentenceEmbedder,
        val: DialogueDataset = None,
        clustering: tp.Optional[OneViewClustering] = None,
        n_clusters: int = 60,
        n_clusters_first_stage: int = 400,
        separator: tp.Optional[tp.Callable[[Utterance, Dialogue], tp.Hashable]] = speaker_filter,
        sep_before: bool = True
) -> FrequencyDialogueGraph:
    """Build dialogue graph in two stages.

    Args:
        train:
        val:
        embedder:
        clustering: pre-computed clustering of n_clusters_first_stage size
        n_clusters: final number of clusters
        n_clusters_first_stage: number of clusters to split into during the first stage
        separator: function that extracts group identificator from utterances, the default one returns 1 for system utterances and 0 for user utterances
        sep_before:

    Returns:

    """
    if clustering is not None:
        assert clustering.get_nclusters() == n_clusters_first_stage
    logging.info(f"Building graph with params: n_clusters={n_clusters}; n_clusters_first_stage={n_clusters_first_stage}")
    (
        graph_merged,
        cluster_embeddings,
        cluster_kmeans_labels,
        clustering_merged,
    ) = full_merge_build(
        train,
        embedder,
        clustering or clusterize(train, embedder, n_clusters=n_clusters_first_stage),
        val,
        n_clusters,
        separator=separator,
        sep_before=sep_before,
    )

    logging.info("Graph built.")
    return graph_merged
