from df_graph_construction import __version__
import logging
import pickle

import pytest
import typing as tp
from dataclasses import dataclass

from df_graph_construction.cli import gc
from df_graph_construction.dataset import DialogueDataset, Dialogue
from df_graph_construction.embedders import SentenceEmbedder, OneViewEmbedder
from df_graph_construction.dialogue_graph.dgac import dgac_one_stage, dgac_two_stage
from df_graph_construction.dialogue_graph import FrequencyDialogueGraph
from df_graph_construction.clustering.filters import speaker_filter

from datasets import load_dataset
import json


def test_version():
    assert __version__ == "0.1.0"


@dataclass
class Dataset:
    name: str
    train: DialogueDataset
    test: DialogueDataset
    val: DialogueDataset


@pytest.fixture(scope="function")
def embedder() -> OneViewEmbedder:
    logging.info("Creating embedder")
    return SentenceEmbedder(device=0)


@pytest.fixture(scope="module")
def dgac_dataset(request: pytest.FixtureRequest) -> tp.Callable[[str], Dataset]:
    def _dgac_dataset(name: str) -> Dataset:
        logging.info(f"Loading dataset {name}")
        dataset = load_dataset(name)
        test = DialogueDataset.from_dataset(dataset['test'])
        val = DialogueDataset.from_dataset(dataset['validation'])
        train = DialogueDataset.from_dataset(dataset['train'])

        logging.info(f"Dataset {name} loaded")

        return Dataset(
            name=name,
            train=train,
            test=test,
            val=val
        )
    return _dgac_dataset


@pytest.mark.parametrize("dataset_name", ["multi_woz_v22", "schema_guided_dstc8"])
class TestDGAC:
    def test_one_stage(
            self, dgac_dataset: tp.Callable[[str], Dataset], embedder: OneViewEmbedder, dataset_name: str
    ) -> None:
        loaded_dataset = dgac_dataset(dataset_name)
        graph_one_stage = dgac_one_stage(loaded_dataset.train, embedder=embedder)
        _, _, _, one_stage_accs = graph_one_stage.success_rate(loaded_dataset.test, acc_ks=[3, 5, 10])

        logging.info(f"{loaded_dataset.name} accuracy: {one_stage_accs}")
        assert max(one_stage_accs) > 0.7 and min(one_stage_accs) > 0.3

    def test_two_stage(
            self, dgac_dataset: tp.Callable[[str], Dataset], embedder: OneViewEmbedder, dataset_name: str
    ) -> None:
        loaded_dataset = dgac_dataset(dataset_name)
        graph_two_stage = dgac_two_stage(loaded_dataset.train, embedder=embedder, separator=speaker_filter)
        _, _, _, two_stage_accs = graph_two_stage.success_rate(loaded_dataset.test, acc_ks=[3, 5, 10])

        logging.info(f"{loaded_dataset.name} accuracy: {two_stage_accs}")
        assert max(two_stage_accs) > 0.7 and min(two_stage_accs) > 0.3


@pytest.mark.parametrize("graph_factory", [
    lambda train, embedder: dgac_one_stage(train, embedder),
    lambda train, embedder: dgac_two_stage(train, embedder=embedder, separator=speaker_filter, sep_before=True),
])
def test_clusterization(
        embedder,
        dgac_dataset: tp.Callable[[str], Dataset],
        graph_factory: tp.Callable[[DialogueDataset, OneViewEmbedder], FrequencyDialogueGraph],
) -> None:
    train = dgac_dataset("multi_woz_v22").train
    graph = graph_factory(train, embedder)
    dialogues = load_dataset("schema_guided_dstc8")["test"]
    for d_id, dialogue in enumerate(dialogues):
        for idx, \
            (speaker,
             (cluster, emb)) in \
                enumerate(zip(
                    dialogue["turns"]["speaker"],
                    graph.iter_dialogue(Dialogue.from_dataset(dialogue))
                )):
            pred = train.get_utterance_by_idx(graph.clustering.get_cluster(cluster).utterances[0]).speaker
            assert speaker == pred


@pytest.mark.parametrize("pretrained_model,dataset_from_huggingface,dataset_from_file", [
    (None, "multi_woz_v22", None),
    (None, None, "examples/example_dataset.dialogues")
])
def test_main(pretrained_model, dataset_from_huggingface, dataset_from_file, tmp_path) -> None:
    gc(
        example_dialogues="examples/example_dialogues",
        output_file=tmp_path / "output.json",
        pretrained_model=pretrained_model,
        dataset_from_huggingface=dataset_from_huggingface,
        dataset_from_file=dataset_from_file,
        save_pretrained=tmp_path / "model.graph",
    )

    with open(tmp_path / "output.json", "r") as f:
        result = json.load(f)

    assert result.get("start_node") is not None

    with open(tmp_path / "model.graph", "rb") as f:
        graph: FrequencyDialogueGraph = pickle.load(f)

    assert graph.clustering.get_nclusters() == 60

    gc(
        example_dialogues="examples/example_dialogues",
        output_file=tmp_path / "output.json",
        pretrained_model=tmp_path / "model.graph",
        dataset_from_huggingface=None,
        dataset_from_file=None,
        save_pretrained=None,
    )

    with open(tmp_path / "output.json", "r") as f:
        result = json.load(f)

    assert result.get("start_node") is not None

