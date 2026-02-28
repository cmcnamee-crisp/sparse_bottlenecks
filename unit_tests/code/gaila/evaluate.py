from typing import Any, Dict, List
import numpy as np
import torch as th

import helpers
import flags


def get_predictions(
    FLAGS,
    dataspec,
    datamodule,
    trainer,
    model,
    num_classes,
    intervene_dataspec,
    info=None,
) -> List[Dict[str, Any]]:
    """Predictions per class. Used for constructing confusion matrices.
    """
    if info is None:
        info = {}
    out = []

    # classname -> classlabel
    class2label = helpers.dataspec_to_mapping(dataspec)
    class2concept = {cls: spec["conceptname"] for cls, spec in dataspec.items()}
    label2concept = {
        spec["classlabel"]: spec["conceptname"] for spec in dataspec.values()
    }

    with th.no_grad():
        out = []
        for classname_test, dataloader in datamodule.get_constituent_dataloaders():
            # TODO: Refactor for constituents.
            _holdout_name = {False: "seen", True: "unseen"}[
                intervene_dataspec[classname_test]["holdout"]
            ]
            clss_out = {
                **flags.namespace_to_dict(FLAGS),
                "classname": classname_test,
                "classlabel": class2label[classname_test],
                "conceptname": class2concept[classname_test],
                "holdout": _holdout_name,
                **{k: 0 for k in class2concept.values()},
                **info,
            }
            preds = th.cat(
                trainer.predict(model=model, dataloaders=[dataloader])
            ).argmax(1)
            clss_out["accuracy"] = (
                (preds == class2label[classname_test]).float().mean().item()
            )
            _predictions = np.bincount(
                preds.cpu().numpy(), minlength=num_classes,  # 18
            )
            try:
                for j in range(len(_predictions)):
                    clss_out[label2concept[j]] += _predictions[j]
            except:
                exit()
            out.append(clss_out)
    return out


def get_accuracy(
    FLAGS, trainer, dataspec, datamodule, model, info=None
) -> List[Dict[str, Any]]:
    """Accuracy per class.
    """
    if info is None:
        info = {}

    out = []
    with th.no_grad():
        model.eval()
        for classname_test, dataloader in datamodule.get_constituent_dataloaders():
            test_result = trainer.test(model=model, test_dataloaders=[dataloader])
            _holdout_name = {False: "seen", True: "unseen"}[
                dataspec[classname_test]["holdout"]
            ]
            out.append(
                {
                    **flags.namespace_to_dict(FLAGS),
                    "classname": classname_test,
                    "accuracy": test_result[0]["test_acc"],
                    "holdout": _holdout_name,
                    **info,
                }
            )
    return out
