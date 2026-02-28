"""Generates minimal pairs over samples.
"""
import argparse
import tqdm
import time
import os

import numpy as np
import pandas as pd

from pytorch_lightning import seed_everything

import dataset


def main(FLAGS):
    seed_everything(FLAGS.seed)
    n_test_per = FLAGS.n_test
    os.makedirs(f"data/images/{FLAGS.data_path}", exist_ok=True)
    out = []
    counter = 0
    samples = [
        dataset.generate_sample(
            # color = black.
            (0, 0, 0)
        )
        for _ in range(n_test_per)
    ]

    if FLAGS.colors:
        COLOR_VARIANCE = 10
        colors = [dataset.Color(*c, COLOR_VARIANCE).mean() for c in dataset.COLORS]
    else:
        colors = [(0, 0, 0)]
    with tqdm.tqdm(total=n_test_per * 18 * len(colors), desc="Generating") as pbar:
        for (
            concept_idx,
            (concept, sname, shape, lname, layout, rname, stroke),
        ) in enumerate(dataset.concepts()):
            for counterfactual_idx, sample in enumerate(samples):
                for color_idx, color in enumerate(colors):
                    s = dataset.Sample(*sample[:-1], color=color)
                    instance = dataset.draw(
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
                        s,
                        output=FLAGS.data_path,
                        output_name=f"{counter}_{concept}_{FLAGS.seed}_{color_idx}",
                        info={
                            "counterfactual_idx": counterfactual_idx,
                            "color_idx": color_idx,
                            "classlabel": concept_idx,
                        },
                    )
                    out.append(instance)
                    counter += 1
                    pbar.update()
    df = pd.DataFrame(out)
    df["image_id"] = np.arange(len(df))
    df.to_csv(
        f"data/datadesc_{FLAGS.data_path}_{FLAGS.seed}.tsv", index=False, sep="\t"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--jobname", type=str)
    parser.add_argument(
        "--data_path",
        type=str,
        default="t1",
        help="Meant for either mturk/t1/t1colors.",
    )
    parser.add_argument("--n_test", type=int, default=10)
    parser.add_argument("--colors", type=bool, default=False)
    FLAGS = parser.parse_args()

    tick = time.time()
    main(FLAGS)
    tock = time.time()
    pd.DataFrame(
        [
            {
                "id": "dataset_t1",
                "script": "dataset_t1",
                "seconds": tock - tick,
                "seed": FLAGS.seed,
            }
        ]
    ).to_csv(f"times/dataset_t1_{FLAGS.seed}.tsv", index=False, sep="\t")
