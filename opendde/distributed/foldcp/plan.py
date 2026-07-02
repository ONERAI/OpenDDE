"""Static Fold-CP migration task plan used by tests and documentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoldCPTask:
    task_id: str
    title: str
    open_dde_stage: str
    foldcp_source: str
    expected_memory_effect: str
    validation: str


FOLDCP_TASKS: tuple[FoldCPTask, ...] = (
    FoldCPTask(
        "task0",
        "flags, metrics, reports, and launch contract",
        "runner/config",
        "Fold-CP paper Section 3.1 setup; Boltz DistributedManager docs",
        "No model memory reduction yet; establishes measurement contract.",
        "pytest tests/foldcp_tasks/test_foldcp_preflight.py",
    ),
    FoldCPTask(
        "task1",
        "2D CP mesh and Ring2DComm base primitives",
        "distributed bootstrap before model forward",
        "Fold-CP square topology; boltz.distributed.comm.Ring2DComm",
        "Prepares N^2 shard ownership; no core layer reduction until task2.",
        "4-rank UT validates row/column/transpose ownership.",
    ),
    FoldCPTask(
        "task2",
        "pair tensor sharding at dataloader/to_device boundary",
        "feature transfer into CUDA",
        "Fold-CP spatial partitioning over 2D pair tensors",
        "Large z/pair masks move from full N^2 per GPU to about N^2/4.",
        "Single vs CP gathered tensors bitwise equal.",
    ),
    FoldCPTask(
        "task3",
        "outer product mean ring computation",
        "MSA module writes MSA information back to pair z",
        "Fold-CP pair construction; boltz distributed outer_product_mean",
        "Avoids materializing full O(N_msa*N^2) intermediates on one GPU.",
        "FP32/FP64 reference histogram and gathered output equality thresholds.",
    ),
    FoldCPTask(
        "task4",
        "triangle multiplication outgoing/incoming",
        "PairformerBlock tri_mul_out and tri_mul_in",
        "Fold-CP ring/Cannon-style distributed BMM",
        "Major Pairformer N^2 activation and BMM workspace reduction.",
        "Bitwise where kernels match; otherwise max_abs/max_rel vs serial.",
    ),
    FoldCPTask(
        "task5",
        "triangle attention with ring online softmax",
        "PairformerBlock tri_att_start and tri_att_end",
        "Fold-CP tiled/online softmax attention update",
        "Cuts attention logits/bias from full N^2 per GPU to sharded tiles.",
        "Online softmax UT plus serial-vs-CP output comparison.",
    ),
    FoldCPTask(
        "task6",
        "PairformerStack end-to-end CP wiring",
        "all trunk pair blocks",
        "Fold-CP chapter 3 integration of distributed pair modules",
        "First trunk-level practical peak-memory reduction across repeated blocks.",
        "512 speed/memory regression plus 1536/2048 capacity probe.",
    ),
    FoldCPTask(
        "task7",
        "MSA pair weighted averaging CP path",
        "MSAStack weighted average over pair z",
        "Fold-CP attention-style online reduction over pair bias",
        "Reduces large pair-weighted MSA attention buffers.",
        "Serial-vs-CP MSA update equality and stage peak reduction.",
    ),
    FoldCPTask(
        "task8",
        "structural token pair expansion CP path",
        "StructuralTokenExpander gather_parent_pair and role pair features",
        "Fold-CP 2D pair partitioning applied to structural-token pair tensors",
        "Reduces full structural N_struct^2 pair activation allocation.",
        "Gathered structural z and pair bias match serial reference.",
    ),
    FoldCPTask(
        "task9",
        "atom/window CP path and final capacity benchmark",
        "diffusion atom encoder/decoder plus full inference",
        "Fold-CP atom/token gather, window ownership, and full prediction workflow",
        "Targets large O(N_atom) buffers and confirms the largest four-card length.",
        "Window ownership UT, 512 timing, and 2048/3072 capacity probes.",
    ),
)


def plan_markdown_table() -> str:
    rows = [
        "| Task | Stage | Fold-CP content | Expected memory effect | Validation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for task in FOLDCP_TASKS:
        rows.append(
            f"| {task.task_id}: {task.title} | {task.open_dde_stage} | "
            f"{task.foldcp_source} | {task.expected_memory_effect} | "
            f"{task.validation} |"
        )
    return "\n".join(rows)
