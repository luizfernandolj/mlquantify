from dataclasses import dataclass, field


@dataclass
class TargetInputTags:

    one_d: bool = True
    two_d: bool = False
    continuous: bool = False
    multi_class: bool = True
    multi_label: bool = False


@dataclass
class Tags:
    
    estimator: bool = False
    estimation_type: str | None
    estimator_type: str | None
    sampler: str | None
    aggregation_type: str | None
    target_input_tags: TargetInputTags



def get_tags(quantifier):
    try:
        tags = quantifier.__mlquantify_tags__()
    except AttributeError as ext:
        if "has no attribute '__mlquantify_tags__'" in str(ext):
            raise AttributeError("Quantifier is missing __mlquantify_tags__ method, ensure your class is inheriting from BaseQuantifier.")
        else:
            raise
    return tags