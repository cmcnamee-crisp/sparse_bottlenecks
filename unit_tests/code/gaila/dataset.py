"""Primary script for generating the dataset.

To generate instances with various colors, use `--color True` and `--p [float]`
to correlate each instance with an arbirtarily chosen (fixed) color with rate `p`.
"""
from typing import Any, Dict, List, Tuple, Union

Function = Any
import argparse
import time
import random
import math
import os
import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, RegularPolygon
import matplotlib.patheffects as path_effects

import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything


from collections import namedtuple


class Color:
    def __init__(self, R, G, B, variance) -> None:
        # assumes an interger 0..255
        self.R = int(R)
        self.G = int(G)
        self.B = int(B)

        self.variance = variance

    def mean(self):
        return (self.R / 255, self.G / 255, self.B / 255)

    def sample(self):
        clip = lambda x: int(np.clip(x, 0, 255)) / 255
        return (
            clip(np.random.normal(self.R, self.variance)),
            clip(np.random.normal(self.G, self.variance)),
            clip(np.random.normal(self.B, self.variance)),
        )


COLORS = [
    (240, 120, 240),
    (240, 240, 5),
    (5, 240, 120),
    (5, 5, 120),
    (120, 5, 240),
    (240, 5, 120),
    (5, 240, 240),
    (120, 5, 5),
    (240, 5, 5),
    (120, 240, 120),
    (120, 120, 5),
    (5, 120, 240),
    (5, 240, 5),
    (240, 240, 120),
    (120, 5, 120),
    (240, 120, 5),
    (240, 240, 240),
    (5, 5, 240),
]

Sample = namedtuple(
    "Sample", ["k", "w", "h", "r", "i", "j", "noise", "noise_2", "color"]
)
"""The statistics necessary to draw an image of a given concept.

k: int, number of shapes
w: List[float], list of shape widths
h: List[float], list of shape heights
r: List[float], list of shape radius-es
i: float, the center
j: List[float], the deltas about that center
noise: List[noise], noise used to jostle the concepts
@unused
noise_2: List[noise], noise used to jostle the concepts
color
"""

# Global Variables
CANVAS_SIZE = 15

# size of shapes
MEAN_SHAPE_WIDTH = 4
SHAPE_WIDTH_VAR = 1

# where shape is centered on canvas
MEAN_CENTER = 0
CENTER_VAR = 2

# vertical/horizontal range that shapes occupy
RANGE = 10

# number of shapes
MEAN_SHAPE = MEAN_SHAPES = 10
SHAPE_VAR = 2

# variation from center/layout skeleton
MEAN_NOISE = 2
NOISE_VAR = 1

COLOR = "black"
FILL = True
LW = 10  # linewidth

# path effects for fuzzy stroke style
fuzz = [path_effects.withTickedStroke(offset=(0, 0), length=1.5, spacing=5, angle=-15)]

plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True

CONCEPT2IDX = {
    "slup": 0,
    "wug": 1,
    "wok": 2,
    "gurp": 3,
    "bix": 4,
    "blorp": 5,
    "surp": 6,
    "gix": 7,
    "blug": 8,
    "wix": 9,
    "gip": 10,
    "dok": 11,
    "dax": 12,
    "blick": 13,
    "glorp": 14,
    "boop": 15,
    "bip": 16,
    "glick": 17,
}


def generate_color(P: float, colors: List[Color], concept: str):
    """Generate a color for the given concept.
    """
    class_idx = CONCEPT2IDX[concept]
    if random.random() < P:
        color_idx = class_idx
    else:
        color_idx = random.randint(0, len(colors) - 1)
        while color_idx == class_idx:
            color_idx = random.randint(0, len(colors) - 1)
    color = colors[color_idx].sample()
    return color, color_idx


def main(FLAGS):
    seed_everything(FLAGS.seed)
    n_train_per = FLAGS.n_train
    n_test_per = FLAGS.n_test

    if FLAGS.colors:
        data_path = f"{FLAGS.data_path}_{FLAGS.color_corr}"
    else:
        data_path = FLAGS.data_path
    os.makedirs(f"data/images/{data_path}", exist_ok=True)

    counter = 0
    out = []

    if FLAGS.colors:
        COLOR_VARIANCE = 10
        colors = [Color(*c, COLOR_VARIANCE) for c in COLORS]

    with tqdm.tqdm(total=18 * (n_train_per + n_test_per), desc="Generating") as pbar:
        for (
            concept_idx,
            (concept, sname, shape, lname, layout, rname, stroke),
        ) in enumerate(concepts()):
            for _ in range(n_train_per):
                if FLAGS.colors:
                    color, color_idx = generate_color(FLAGS.color_corr, colors, concept)
                else:
                    color, color_idx = (0, 0, 0), -1  # black
                sample = generate_sample(color)
                instance = draw(
                    counter,
                    concept,
                    sname,
                    lname,
                    rname,
                    shape,
                    layout,
                    stroke,
                    "train",
                    FLAGS.seed,
                    sample,
                    data_path,
                    f"{counter}_{concept}_{FLAGS.seed}",
                    info={"classlabel": concept_idx, "color_idx": color_idx},
                )
                out.append(instance)
                counter += 1
                pbar.update()
            for _ in range(n_test_per):
                if FLAGS.colors:
                    color, color_idx = generate_color(FLAGS.color_corr, colors, concept)
                else:
                    color, color_idx = (0, 0, 0), -1  # black
                sample = generate_sample(color)
                instance = draw(
                    counter,
                    concept,
                    sname,
                    lname,
                    rname,
                    shape,
                    layout,
                    stroke,
                    "test",
                    FLAGS.seed,
                    sample,
                    data_path,
                    f"{counter}_{concept}_{FLAGS.seed}",
                    info={"classlabel": concept_idx, "color_idx": color_idx},
                )
                out.append(instance)
                counter += 1
                pbar.update()

    df = pd.DataFrame(out)
    df["image_id"] = np.arange(len(df))
    df.to_csv(f"data/datadesc_{data_path}_{FLAGS.seed}.tsv", index=False, sep="\t")


def concepts() -> List:
    """Produces a list of concepts. """
    layout_names, layouts, shape_names, shapes, stroke_names, strokes = _consituents()
    concept_names = [
        "slup",
        "wug",
        "wok",
        "gurp",
        "bix",
        "blorp",
        "surp",
        "gix",
        "blug",
        "wix",
        "gip",
        "dok",
        "dax",
        "blick",
        "glorp",
        "boop",
        "bip",
        "glick",
    ]
    concept_idx = 0
    out = []
    for lname, layout in zip(layout_names, layouts):
        for rname, stroke in zip(stroke_names, strokes):
            for sname, shape in zip(shape_names, shapes):
                concept = concept_names[concept_idx]
                concept_idx += 1
                out.append((concept, sname, shape, lname, layout, rname, stroke))
    return out


def _consituents() -> Tuple[List]:
    layout_names = ["row", "column", "ring"]
    layouts = [vertical_layout, horizontal_layout, circle_layout]
    shape_names = ["oval", "rectangle", "poly"]
    shapes = [draw_oval, draw_rectangle, draw_poly]
    stroke_names = ["clean", "fuzzy"]
    strokes = [None, fuzz]
    return layout_names, layouts, shape_names, shapes, stroke_names, strokes


def draw(
    counter: int,
    concept: str,
    sname: str,
    lname: str,
    rname: str,
    shape: Function,
    layout: Function,
    stroke: Function,
    split: str,
    seed: int,
    sample: Sample,
    output: str,
    output_name: str,
    info=None,
) -> Dict[str, Union[int, str]]:
    """Draws a sample of the given concept based on the statistics provided by the sample.
    
    `info` tracks additional information to be stored in the row.
    """
    if info is None:
        # default arguments should be immutable.
        info = {}
    _, axes = plt.subplots(1, 1, figsize=(2, 2))
    gen_concept(axes, layout, shape, stroke, sample)
    axes.set_axis_off()
    plt.savefig(f"data/images/{output}/{output_name}.png", bbox_inches="tight")
    plt.close()
    return {
        "filename": f"{output_name}.png",
        "concept": concept,
        "classname": f"{lname}-{sname}-{rname}",
        "layout": lname,
        "shape": sname,
        "stroke": rname,
        "test": split == "test",
        "seed": seed,
        **info,
    }


def generate_sample(color) -> Sample:
    """Generates a `Sample` based on the static constants in the namespace.
    """
    k = 0
    while k < 1:
        k = int(round(np.random.normal(MEAN_SHAPES, SHAPE_VAR)))
    w = [np.random.normal(MEAN_SHAPE_WIDTH, SHAPE_WIDTH_VAR) for _ in range(k)]
    h = [np.random.normal(MEAN_SHAPE_WIDTH, SHAPE_WIDTH_VAR) for _ in range(k)]
    r = RANGE / 2 + np.random.normal(MEAN_NOISE, NOISE_VAR)
    i = np.random.normal(MEAN_CENTER, CENTER_VAR)
    j = [random.uniform(-RANGE, RANGE) for _ in range(k)]
    noise = [np.random.normal(MEAN_NOISE, NOISE_VAR) for _ in range(k)]
    noise_2 = [np.random.normal(MEAN_NOISE, NOISE_VAR) for _ in range(k)]
    return Sample(
        **{
            "k": k,
            "w": w,
            "h": h,
            "r": r,
            "i": i,
            "j": j,
            "noise": noise,
            "noise_2": noise_2,
            "color": color,
        }
    )


def vertical_layout(k, sample):
    """generate centers in a roughly horizontal line"""
    x = sample.i
    centers = []
    for idx in range(k):
        centers.append((x + sample.noise[idx], sample.j[idx]))
    return centers


def horizontal_layout(k, sample):
    """generate centers in a roughly vertical line"""
    y = sample.i
    centers = []
    for idx in range(k):
        centers.append((sample.j[idx], y + sample.noise[idx]))
    return centers


def circle_layout(k, sample):
    """generate centers in a roughly circular pattern"""

    r = sample.r
    centers = []
    for idx in range(k):
        angle = random.uniform(-10, 10) * math.pi * 2
        x = math.cos(angle) * r
        y = math.sin(angle) * r
        x += sample.noise[idx]
        y += sample.noise[idx]
        centers.append((x, y))
    return centers


def draw_poly(ax, center, stroke=None, sample=None, idx=-1):
    assert sample
    r = (sample.h[idx] + sample.w[idx]) / 4.0
    num_verts = random.uniform(3, 7)
    rotation = random.uniform(0, 6)  # in radians
    if stroke:
        circle = RegularPolygon(
            xy=center,
            numVertices=int(num_verts),
            radius=r,
            orientation=rotation,
            fill=FILL,
            linewidth=LW,
            edgecolor=sample.color,
            facecolor=sample.color,
            path_effects=stroke,
        )
    else:
        circle = RegularPolygon(
            xy=center,
            numVertices=int(num_verts),
            radius=r,
            orientation=rotation,
            fill=FILL,
            linewidth=LW,
            edgecolor=sample.color,
            facecolor=sample.color,
        )
    ax.add_patch(circle)


def draw_rectangle(ax, center, stroke=None, sample=None, idx=-1):
    assert sample
    w = sample.w[idx]
    h = sample.h[idx]
    if stroke:
        rec = Rectangle(
            center,
            w,
            h,
            fill=FILL,
            linewidth=LW,
            edgecolor=sample.color,
            facecolor=sample.color,
            path_effects=stroke,
        )
    else:
        rec = Rectangle(
            center,
            w,
            h,
            fill=FILL,
            linewidth=LW,
            edgecolor=sample.color,
            facecolor=sample.color,
        )
    ax.add_patch(rec)


def draw_oval(ax, center, stroke=None, sample=None, idx=-1):
    assert sample
    w = sample.w[idx]
    h = sample.h[idx]
    if stroke:
        circle = Ellipse(
            center,
            w,
            h,
            fill=FILL,
            linewidth=LW,
            edgecolor=sample.color,
            facecolor=sample.color,
            path_effects=stroke,
        )
    else:
        circle = Ellipse(
            center,
            w,
            h,
            fill=FILL,
            linewidth=LW,
            edgecolor=sample.color,
            facecolor=sample.color,
        )
    ax.add_patch(circle)


def gen_concept(ax, layout, shape, stroke, sample) -> None:
    k = sample.k
    centers = layout(k, sample)
    for idx, c in enumerate(centers):
        shape(ax, c, stroke, sample, idx)
    ax.set_xlim(-CANVAS_SIZE, CANVAS_SIZE)
    ax.set_ylim(-CANVAS_SIZE, CANVAS_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


if __name__ == "__main__":
    tick = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=6, type=int)
    parser.add_argument("--jobname", type=str)
    parser.add_argument("--data_path", type=str, default="default")
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_train", type=int, default=1000)

    parser.add_argument("--colors", type=bool, default=False)
    parser.add_argument("--color_corr", type=float, default=1.0)

    FLAGS = parser.parse_args()
    main(FLAGS)
    tock = time.time()
    pd.DataFrame(
        [
            {
                "id": "dataset",
                "script": "dataset",
                "seconds": tock - tick,
                "seed": FLAGS.seed,
            }
        ]
    ).to_csv(f"times/dataset_{FLAGS.seed}.tsv", index=False, sep="\t")
