"""
RL 数据加载与提示词构建：加载/切分 backgrounds、构建场景初始状态、格式化对话为模型输入。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from utils import (
    build_new_system_prompt_with_partner,
    format_conversation_to_prompt,
    load_backgrounds,
)


def load_and_split_backgrounds(
    backgrounds_path: Optional[str],
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    从 backgrounds 文件加载场景，打乱后按 (1 - val_ratio) / val_ratio 划分为 train / val。
    若 backgrounds_path 为空或未提供，返回 ([], [])。

    Returns:
        (train_scenes, val_scenes)
    """
    if not backgrounds_path:
        return [], []

    backgrounds = load_backgrounds(backgrounds_path)
    if not backgrounds:
        return [], []

    rng = random.Random(seed if seed is not None else 0)
    rng.shuffle(backgrounds)
    split_idx = int(len(backgrounds) * (1.0 - float(val_ratio)))
    split_idx = max(1, min(split_idx, len(backgrounds)))  # ensure at least one train scene
    train_scenes = backgrounds[:split_idx]
    val_scenes = backgrounds[split_idx:]
    return train_scenes, val_scenes



def build_scene_initial_state_with_partner_prompt(
    scene: dict,
) -> Tuple[str, str, Dict[str, List[Dict[str, str]]]]:
    """
    使用包含 partner profile 的新系统提示词构建初始状态。

    若 scene 含 `_swap_agents=True`，则交换 a0 / a1 的角色绑定。
    搭配 `expand_scenes_with_role_swap`，可让 policy / partner 在同一场景中互换扮演位置，
    避免 policy 模型只学习扮演单一角色。
    """
    agents_bg = scene.get("agents_background") or {}
    agent_names_scene = list(agents_bg.keys())
    if bool(scene.get("_swap_agents", False)) and len(agent_names_scene) >= 2:
        a0, a1 = agent_names_scene[1], agent_names_scene[0]
    else:
        a0, a1 = agent_names_scene[0], agent_names_scene[1]
    goal0 = (scene.get("social_goals") or {}).get(a0)
    goal1 = (scene.get("social_goals") or {}).get(a1)
    sys0 = build_new_system_prompt_with_partner(
        a0,
        agents_bg.get(a0, ""),
        agents_bg.get(a1, ""),
        goal0,
        scene.get("scenario"),
    )
    sys1 = build_new_system_prompt_with_partner(
        a1,
        agents_bg.get(a1, ""),
        agents_bg.get(a0, ""),
        goal1,
        scene.get("scenario"),
    )
    history = {
        a0: [{"role": "system", "content": sys0}, {"role": "user", "content": "you speak first."}],
        a1: [{"role": "system", "content": sys1}],
    }
    return a0, a1, history


def expand_scenes_with_role_swap(
    scenes: List[dict],
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    对每个场景生成两个变体：
    - `_swap_agents=False`：原始角色绑定（policy 扮演原始 agent1，partner 扮演原始 agent0）。
    - `_swap_agents=True` ：互换角色绑定（policy 扮演原始 agent0，partner 扮演原始 agent1）。

    这样在整个数据集上 policy 模型在每个场景里都会扮演两种角色，避免训练 / 测试时只
    学到单边视角。可选 shuffle 以便 DDP 各 rank 上原始 / 交换样本分布均衡。
    """
    if not scenes:
        return []
    expanded: List[dict] = []
    for scene in scenes:
        base = dict(scene)
        base.pop("_swap_agents", None)
        original = dict(base)
        original["_swap_agents"] = False
        swapped = dict(base)
        swapped["_swap_agents"] = True
        expanded.append(original)
        expanded.append(swapped)
    if shuffle:
        rng = random.Random(seed if seed is not None else 0)
        rng.shuffle(expanded)
    return expanded


def build_batch_initial_states(
    batch_scenes: List[dict],
) -> Tuple[List[str], List[str], List[Dict[str, List[Dict[str, str]]]]]:
    """
    对一批 scene 构建初始状态，用于训练循环的 batch 初始化。

    Returns:
        (agent0_list, agent1_list, history_messages_all)
    """
    agent0_list: List[str] = []
    agent1_list: List[str] = []
    history_messages_all: List[Dict[str, List[Dict[str, str]]]] = []
    for scene in batch_scenes:
        a0, a1, history = build_scene_initial_state_with_partner_prompt(scene)
        agent0_list.append(a0)
        agent1_list.append(a1)
        history_messages_all.append(history)
    return agent0_list, agent1_list, history_messages_all


# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------


def _scene_collate_fn(batch: list) -> list:
    """collate_fn：保持 batch 为 scene dict 的列表，不做 tensor 堆叠。"""
    return list(batch)


class SceneDataset(Dataset):
    """Dataset：每个样本为一个 scene 字典。"""

    def __init__(self, scenes: List[dict]) -> None:
        self.scenes = scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        return self.scenes[idx]


class SceneDataLoader:
    """
    从 backgrounds 文件加载数据并构建 DataLoader。
    迭代时每批得到「scene 字典的列表」，可直接传入 build_batch_initial_states。
    """

    def __init__(
        self,
        backgrounds_path: Optional[str],
        val_ratio: float = 0.15,
        seed: Optional[int] = None,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> None:
        self.backgrounds_path = backgrounds_path
        self.val_ratio = val_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_scenes: List[dict] = []
        self.val_scenes: List[dict] = []
        self._load()

    def _load(self) -> None:
        self.train_scenes, self.val_scenes = load_and_split_backgrounds(
            self.backgrounds_path,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:
        ds = SceneDataset(self.train_scenes)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=_scene_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        ds = SceneDataset(self.val_scenes)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_scene_collate_fn,
        )


@dataclass
class DialogueBatchState:
    """单个 batch 在某一轮次的状态与提示词。"""

    batch_idx: int
    round_idx: int
    is_large: bool
    scenes: List[dict]
    agent0_list: List[str]
    agent1_list: List[str]
    history_messages_all: List[Dict[str, List[Dict[str, str]]]]
    last_reply: List[Optional[str]]
    speakers: List[str]
    prompts: List[str]

    def commit_replies(self, replies: List[str]) -> None:
        """将本轮 replies 写回 history，并更新 last_reply。"""
        if len(replies) != len(self.speakers):
            raise ValueError("Replies length must match batch size.")
        for i, reply in enumerate(replies):
            r = (reply or "").strip()
            self.last_reply[i] = r
            speaker = self.speakers[i]
            self.history_messages_all[i][speaker].append(
                {"role": "assistant", "content": r}
            )


class DialogueDataLoader:
    """
    记录对话轮次与上下文的 DataLoader：
    迭代时直接返回可输入模型的 prompts 及上下文信息。
    """

    def __init__(
        self,
        backgrounds_path: Optional[str],
        batch_size: int,
        max_rounds: int,
        which_agent_large: str = "agent0",
        val_ratio: float = 0.15,
        seed: Optional[int] = None,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        train_scenes: Optional[List[dict]] = None,
    ) -> None:
        self.backgrounds_path = backgrounds_path
        self.batch_size = max(1, int(batch_size))
        self.max_rounds = max(1, int(max_rounds))
        self.which_agent_large = which_agent_large
        self.val_ratio = val_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.train_scenes = train_scenes
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"Invalid rank/world_size: rank={self.rank}, world_size={self.world_size}"
            )
        self._load()

    def _load(self) -> None:
        train_scenes = list(self.train_scenes)
        if self.world_size <= 1:
            self.train_scenes = train_scenes
            return

        # Match DistributedSampler behavior:
        # pad by repeating from the beginning so each rank gets equal samples.
        num_samples = (len(train_scenes) + self.world_size - 1) // self.world_size
        total_size = num_samples * self.world_size
        if total_size > len(train_scenes):
            train_scenes = train_scenes + train_scenes[: total_size - len(train_scenes)]
        self.train_scenes = train_scenes[self.rank:total_size:self.world_size]

    @property
    def num_batches(self) -> int:
        if not self.train_scenes:
            return 0
        return (len(self.train_scenes) + self.batch_size - 1) // self.batch_size

    def _iter_batches(self) -> List[List[dict]]:
        scenes = list(self.train_scenes)
        if self.shuffle and scenes:
            rng = random.Random(self.seed)
            rng.shuffle(scenes)
        batches = [
            scenes[i : i + self.batch_size]
            for i in range(0, len(scenes), self.batch_size)
        ]
        return batches

    def __iter__(self):
        batches = self._iter_batches()
        if not batches:
            return

        # 初始化每个 batch 的对话状态
        batch_states = []
        for batch_idx, batch_scenes in enumerate(batches):
            agent0_list, agent1_list, history_messages_all = build_batch_initial_states(
                batch_scenes
            )
            last_reply: List[Optional[str]] = [None] * len(batch_scenes)
            batch_states.append(
                {
                    "batch_idx": batch_idx,
                    "scenes": batch_scenes,
                    "agent0_list": agent0_list,
                    "agent1_list": agent1_list,
                    "history_messages_all": history_messages_all,
                    "last_reply": last_reply,
                }
            )

        for batch_state in batch_states:
            agent0_list = batch_state["agent0_list"]
            agent1_list = batch_state["agent1_list"]
            history_messages_all = batch_state["history_messages_all"]
            last_reply = batch_state["last_reply"]

            for round_idx in range(self.max_rounds):
                is_large = (
                    self.which_agent_large == "agent0" and round_idx % 2 == 0
                ) or (self.which_agent_large == "agent1" and round_idx % 2 == 1)

                speakers: List[str] = []
                prompts: List[str] = []
                for i in range(len(agent0_list)):
                    speaker = agent0_list[i] if round_idx % 2 == 0 else agent1_list[i]
                    speakers.append(speaker)
                    if last_reply[i] is not None:
                        history_messages_all[i][speaker].append(
                            {"role": "user", "content": last_reply[i]}
                        )
                    prompts.append(format_conversation_to_prompt(history_messages_all[i][speaker]))

                yield DialogueBatchState(
                    batch_idx=batch_state["batch_idx"],
                    round_idx=round_idx,
                    is_large=is_large,
                    scenes=batch_state["scenes"],
                    agent0_list=agent0_list,
                    agent1_list=agent1_list,
                    history_messages_all=history_messages_all,
                    last_reply=last_reply,
                    speakers=speakers,
                    prompts=prompts,
                )
