"""Helper function for mapping classes to labels.
"""
import itertools


def get_specification_category(target_category: str, classnames):
    assert target_category in {"classes", "layout", "shape", "stroke"}
    if target_category == "classes":
        return get_specification_classes(classnames)

    concepts = set(
        list(itertools.chain.from_iterable([c.split("-") for c in classnames]))
    )

    layouts = [l for l in ["column", "row", "ring",] if l in concepts]
    shapes = [s for s in ["oval", "rectangle", "poly",] if s in concepts]
    strokes = [s for s in ["clean", "fuzzy"] if s in concepts]

    def is_target(name):
        if target_category == "layout":
            for i, concept in enumerate(sorted(layouts)):
                if concept in name:
                    return i
            assert False, name
        if target_category == "shape":
            for i, concept in enumerate(sorted(shapes)):
                if concept in name:
                    return i
            assert False, name
        if target_category == "stroke":
            for i, concept in enumerate(sorted(strokes)):
                if concept in name:
                    return i
            assert False, name

    def get_concept(name):
        if target_category == "layout":
            for i, concept in enumerate(sorted(layouts)):
                if concept in name:
                    return concept
            assert False, name
        if target_category == "shape":
            for i, concept in enumerate(sorted(shapes)):
                if concept in name:
                    return concept
            assert False, name
        if target_category == "stroke":
            for i, concept in enumerate(sorted(strokes)):
                if concept in name:
                    return concept
            assert False, name

    return {
        cn: {
            "classlabel": is_target(cn),
            "conceptname": get_concept(cn),
            "holdout": False,
        }
        for i, cn in enumerate(sorted(classnames))
    }


def get_specification_classes(classnames):
    classes = {
        cn: {"classlabel": i, "conceptname": cn, "holdout": False}
        for i, cn in enumerate(sorted(classnames))
    }
    return classes


def get_specification_color(classnames):
    # Classlabel should be determined per instance by color_idx.
    # TODO: Care/
    classes = {
        cn: {"classlabel": i, "conceptname": f"color_{i}", "holdout": False}
        for i, cn in enumerate(classnames)
    }
    return classes


def get_label(y, kv):
    for k, v in kv.items():
        if k in y:
            return v
    assert False


def holdout(dataspec, targeted_dataspec, FLAGS):
    assert targeted_dataspec in {"layout", "shape", "stroke"}, targeted_dataspec
    # NOTE: This function is *not* functional; we modify the innards
    # of `dataspec`.
    # Still, it seems best practice to return and override the input
    # dataspec, for clarity.
    def _holdout(x: bool) -> bool:
        if FLAGS.holdout_many:
            return not x
        else:
            return x

    for clss, spec in dataspec.items():
        def matches_flags(*flags_names):
            active_flags = [(f, getattr(FLAGS, f, None)) for f in flags_names]
            active_flags = [f_val for f_name, f_val in active_flags if f_val is not None]
            if not active_flags:
                return False  # No holdout specified
            return all(f_val in clss for f_val in active_flags)

        if targeted_dataspec == "layout":
            is_slice = matches_flags("shape_inlp_or_holdout", "stroke_inlp_or_holdout")
            spec["holdout"] = _holdout(is_slice)
        if targeted_dataspec == "shape":
            is_slice = matches_flags("layout_inlp_or_holdout", "stroke_inlp_or_holdout")
            spec["holdout"] = _holdout(is_slice)
        if targeted_dataspec == "stroke":
            is_slice = matches_flags("shape_inlp_or_holdout", "layout_inlp_or_holdout")
            spec["holdout"] = _holdout(is_slice)
    return dataspec
