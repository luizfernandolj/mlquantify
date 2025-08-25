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
    target_input_tags: TargetInputTags
