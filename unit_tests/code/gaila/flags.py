import argparse


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--finetune_model", default="", type=str)
    parser.add_argument(
        "--pretrain_model",
        default=None,
        type=str,
        help="ClipRes, ClipVit, OrigRes, RawRes, RawCnn",
    )

    parser.add_argument("--samples_per_class", default=1000, type=int)
    parser.add_argument(
        "--layout_inlp_or_holdout",
        default=None,
        type=str,
        help="IF category is layout, then this is the inlp-ed class. ELSE this is the concept to holdout.",
    )
    parser.add_argument("--shape_inlp_or_holdout", default=None, type=str)
    parser.add_argument("--stroke_inlp_or_holdout", default=None, type=str)

    # TODO: IMPL THIS PART for APP.
    parser.add_argument("--data_path", default="default", type=str)
    parser.add_argument("--holdout_many", default=False, type=bool)
    parser.add_argument("--dataspec", default=None, type=str, help="classes")
    parser.add_argument(
        "--jobname",
        default="",
        type=str,
        help="name of job/output folder",
        required=True,
    )

    # @Deprecated
    parser.add_argument("--dataspec_analysis", default=None, type=str)
    # @Deprecated
    parser.add_argument(
        "--datadesc", default=None, type=str, help="default=datadesc_default"
    )
    return parser


def namespace_to_string(ns) -> str:
    return _dict_to_string(vars(ns))


def namespace_to_dict(ns):
    return vars(ns)


def get_canonical_keys():
    """Returns the list of keys defined in the main argument parser."""
    parser = get_flags()
    return [action.dest for action in parser._actions if action.dest != "help"]


def namespace_to_string_with_change(ns, kvs) -> str:
    canonical_keys = get_canonical_keys()
    # Start with canonical keys from the namespace
    out = {k: getattr(ns, k) for k in canonical_keys if hasattr(ns, k)}
    # Apply overrides
    out.update(kvs)
    return _dict_to_string(out)


def get_layers(FLAGS):
    return {
        "ClipRes": [
            "conv_layer1",
            "conv_layer2",
            "conv_layer3",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "attnpool",
        ],
        "ClipVit": [
            "conv_embedding",
            "transformer_0",
            "transformer_1",
            "transformer_2",
            "transformer_3",
            "transformer_4",
            "transformer_5",
            "transformer_6",
            "transformer_7",
            "transformer_8",
            "transformer_9",
            "transformer_10",
            "transformer_11",
        ],
        "OrigRes": ["conv_layer0", "layer1", "layer2", "layer3", "layer4"],
        "RawRes": ["conv_layer0", "layer1", "layer2", "layer3", "layer4"],
        "RawCnn": ["conv_layer0", "layer1", "layer2", "layer3", "fc"],
    }[FLAGS.pretrain_model]


def _dict_to_string(out) -> str:
    def _mb_list_to_string(v) -> str:
        if isinstance(v, list):
            return "@".join(sorted(v))
        else:
            return v

    def _filter(v) -> bool:
        if v:
            return True
        else:
            # filter out empty string, None, empty list, False.
            return False

    def shorten_if_str(v):
        if isinstance(v, str):
            if "colors" in v:
                dataset = v.split("_")[-1]
                return f"c{dataset}"
            elif v in {"RawRes", "RawCnn", "ClipRes", "ClipVit", "OrigRes"}:
                return v
            else:
                return v[:2]
        else:
            return v

    out = {k: _mb_list_to_string(v) for k, v in out.items()}
    out = [f"{k[:2]}:{shorten_if_str(v)}" for k, v in out.items() if _filter(v)]
    return "-".join(out).replace("/", "_")


def train_id(ns):
    return "train-" + namespace_to_string_with_change(
        ns,
        {
            # These were blacklisted before, now they are effectively 
            # blacklisted or overridden to None here.
            "dataspec_analysis": None,
            "jobname": None,
        },
    )


def pretrain_id(ns):
    return "pre" + "train-" + namespace_to_string_with_change(
        ns,
        {
            "dataspec": "classes",
            # This is set by downstream tests to focus in on specific
            "dataspec_analysis": None,
            "ft_post_inlp": None,
            "jobname": None,
            "category_inlp": None,
            "layout_inlp_or_holdout": None,
            "shape_inlp_or_holdout": None,
            "stroke_inlp_or_holdout": None,
            "do_inlp_rand": None,
            "inlp_classifier": None,
        },
    )


def train_layerwise_id(ns, layer):
    return train_id(ns) + f"-{layer}"


def t1_id(ns):
    return "t1-" + namespace_to_string_with_change(
        ns,
        {
            "dataspec": None,
            # This is set by downstream tests to focus in on specific
            "dataspec_analysis": None,
            "ft_post_inlp": None,
            "jobname": None,
            "dataspec": None,
            "layout_inlp_or_holdout": None,
            "shape_inlp_or_holdout": None,
            "stroke_inlp_or_holdout": None,
            "holdout_many": None,
            "do_inlp_rand": None,
            "inlp_classifier": None,
        },
    )


def t2_id(ns):
    assert ns.dataspec == "classes", "Change defaults. See doc strings below."
    return "t2-" + namespace_to_string_with_change(
        ns,
        {
            "ft_post_inlp": None,
            "jobname": None,
            "dataspec": None,
            "do_inlp_rand": None,
            "inlp_classifier": None,
        },
    )


def t_causal_id(ns):
    return "causal-" + namespace_to_string_with_change(
        ns,
        {
            "ft_post_inlp": None,
            "jobname": None,
            "dataspec": None,
            "dataspec_analysis": None,
        },
    )


def t3_id(ns):
    assert ns.dataspec == "classes", "Change defaults. See doc strings below."
    return "t3-" + namespace_to_string_with_change(
        ns, {"ft_post_inlp": None, "jobname": None, "dataspec": None,}
    )


def t4_id(ns):
    assert ns.dataspec == "classes", "Change defaults. See doc strings below."
    return "t4-" + namespace_to_string_with_change(
        ns, {"ft_post_inlp": None, "jobname": None, "dataspec": None,}
    )


def t4_wtmt_id(ns):
    assert ns.dataspec == "classes", "Change defaults. See doc strings below."
    return "wtmt-" + namespace_to_string_with_change(
        ns, {"ft_post_inlp": None, "jobname": None, "dataspec": None,}
    )


def dataset_id(pretrain_model_name, data_path, seed):
    if pretrain_model_name == "default":
        model_name = pretrain_model_name.replace("/", "-")
        return f"{data_path}_{64}"
    else:
        model_name = pretrain_model_name.replace("/", "-")
        return f"{data_path}_{model_name}"


def transformed_id(FLAGS, model_data_path=None):
    if model_data_path is not None:
        return f"./data/encode/{FLAGS.data_path}-{model_data_path}-{FLAGS.pretrain_model}-{FLAGS.seed}"
    return f"./data/encode/{FLAGS.data_path}-{FLAGS.pretrain_model}-{FLAGS.seed}"
