from ..dataset import Dialogue, DialogueDataset
from ..dialogue_graph import FrequencyDialogueGraph
import typing as tp
from ..clustering.interface import Cluster
import numpy as np
from pathlib import Path
import json


class AutoList:
    def __init__(self, front_wrap: str = "", back_wrap: str = ""):
        """Contains a list. Representation of it includes front_wrap at the beginning and back_wrap at the end.

        Examples

            >>>auto_list = AutoList(front_wrap="(", end_wrap=")")
            >>>auto_list.append(1)
            >>>auto_list.append(2)
            >>>print(auto_list)
            ([1, 2])
        """
        self.list = []
        self.front_wrap = front_wrap
        self.back_wrap = back_wrap

    def __str__(self):
        return self.front_wrap + str(self.list) + self.back_wrap

    def __repr__(self):
        return self.front_wrap + repr(self.list) + self.back_wrap

    @classmethod
    def auto_condition(cls, label: int):
        return cls(front_wrap="auto_condition(conds=", back_wrap=f", label={label})")


class Reference(tp.TypedDict):
    name: str
    text: str
    utter_index: int
    clusters: tp.Dict[int, float]


class References(tp.TypedDict):
    response: tp.List[Reference]
    condition: tp.List[Reference]
    label: tp.List[Reference]


class Misc(tp.TypedDict):
    references: References


class Node(tp.TypedDict):
    response: str
    transitions: tp.Dict[str, AutoList]
    misc: Misc


def check_speaker_order(
        speakers: tp.List[int]
) -> bool:
    ground_truth = 0
    for speaker in speakers:
        if speaker != ground_truth:
            return False
        ground_truth = 1 - ground_truth
    return True


def group_by_cluster(
        cluster: Cluster,
        train: DialogueDataset,
) -> tp.Hashable:
    return train.get_utterance_by_idx(cluster.utterances[0]).speaker


def add_dialogue(
        dialogue_path: Path,
        graph: FrequencyDialogueGraph,
        dictionary: tp.Dict[str, Node]
) -> None:
    """Add a dialogue to a dictionary of nodes-responses-transitions.

    Args:
        dialogue_path: A path to a dialogue to add
        graph: Graph used to map dialogue utterances to clusters
        dictionary: Dictionary with nodes as keys and response-transitions as values

    Returns:

    """
    if dictionary.get("start_node") is None:
        dictionary["start_node"] = Node(
            response="auto_response(use_refs=False)",
            transitions={},
            misc=Misc(
                references=References(
                    response=[],
                    condition=[],
                    label=[],
                )
            )
        )

    # load dialogue
    with open(dialogue_path, "r") as f:
        file = json.load(f)
        dialogue = Dialogue.from_dataset(file)

    # prepare dialogue
    clusters = np.array(list(cluster for cluster, emb in graph.iter_dialogue(dialogue)))
    speakers = file["turns"]["speaker"]
    assert check_speaker_order(speakers), "Speaker order should be [0, 1, 0, 1, 0, 1, ...]"
    speakers = np.array(speakers)

    user_clusters = clusters[speakers == 0]
    system_clusters = clusters[speakers == 1]
    previous_system_node = "start_node"

    for idx, (user, system) in enumerate(zip(user_clusters, system_clusters)):
        # node creation

        current_system_node = "cluster_" + str(system)
        if dictionary.get(current_system_node) is None:
            dictionary[current_system_node] = Node(
                response="auto_response(use_refs=False)",
                transitions={},
                misc=Misc(
                    references=References(
                        response=[],
                        condition=[],
                        label=[],
                    )
                ),
            )

        # add transition

        transition = f"auto_label({int(system)})"
        if dictionary[previous_system_node]["transitions"].get(transition) is None:
            dictionary[previous_system_node]["transitions"][transition] = AutoList.auto_condition(int(system))
        transition_conditions = dictionary[previous_system_node]["transitions"][transition]
        if int(user) not in transition_conditions.list:
            transition_conditions.list.append(int(user))

        # add misc

        dictionary[current_system_node]["misc"]["references"]["response"].append(Reference(
            name=str(dialogue_path),
            text=dialogue.utterances[idx * 2 + 1].utterance,
            utter_index=idx * 2 + 1,
            clusters={
                int(system): 1.
            }
        ))

        dictionary[previous_system_node]["misc"]["references"]["condition"].append(Reference(
            name=str(dialogue_path),
            text=dialogue.utterances[idx * 2].utterance,
            utter_index=idx * 2,
            clusters={
                int(user): 1.
            }
        ))

        dictionary[previous_system_node]["misc"]["references"]["label"].append(Reference(
            name=str(dialogue_path),
            text=dialogue.utterances[idx * 2 + 1].utterance,
            utter_index=idx * 2 + 1,
            clusters={
                int(system): 1.
            }
        ))

        # cycle

        previous_system_node = current_system_node
    return
