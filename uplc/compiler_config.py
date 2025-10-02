from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class CompilationConfig:
    constant_folding_keep_traces: Optional[bool] = None
    constant_folding: Optional[bool] = None
    unique_variable_names: Optional[bool] = None
    remove_force_delay: Optional[bool] = None
    fold_apply_lambda_increase: Optional[Union[int, float]] = None
    deduplicate: Optional[bool] = None

    def update(
        self, other: Optional["CompilationConfig"] = None, **kwargs
    ) -> "CompilationConfig":
        own_dict = self.__dict__
        other_dict = other.__dict__ if isinstance(other, CompilationConfig) else kwargs
        return self.__class__(
            **{
                k: other_dict.get(k) if other_dict.get(k) is not None else own_dict[k]
                for k in own_dict
            }
        )


# The default configuration for the compiler
OPT_O0_CONFIG = CompilationConfig()
OPT_O1_CONFIG = OPT_O0_CONFIG.update(remove_force_delay=True)
OPT_O2_CONFIG = OPT_O1_CONFIG.update(
    constant_folding=True,
    fold_apply_lambda_increase=1,
    constant_folding_keep_traces=True,
)
OPT_O3_CONFIG = OPT_O2_CONFIG.update(
    deduplicate=True, constant_folding_keep_traces=False
)
OPT_CONFIGS = [OPT_O0_CONFIG, OPT_O1_CONFIG, OPT_O2_CONFIG, OPT_O3_CONFIG]

DEFAULT_CONFIG = CompilationConfig(unique_variable_names=True).update(OPT_O2_CONFIG)

ARGPARSE_ARGS = {
    "unique_variable_names": {
        "__alts__": ["--unique-varnames"],
        "help": "Assign variables a unique name. Some optimizations require this and will be disabled if this is not set.",
    },
    "constant_folding": {
        "__alts__": ["--cf"],
        "help": "Enables experimental constant folding, including propagation and code execution.",
    },
    "constant_folding_keep_traces": {
        "help": "Do not remove traces from the compiled contract during constant folding.",
    },
    "remove_force_delay": {
        "__alts__": ["--rfd"],
        "help": "Removes delayed terms that are immediately forced.",
    },
    "fold_apply_lambda_increase": {
        "__alts__": ["--ala"],
        "help": "Applies terms to lambdas at compile time. The parameter controls how much larger the resulting term is allowed to be. Default is 1, i.e., at most 100% of the original size. Set to 0 to disable.",
        "type": float,
    },
    "deduplicate": {
        "__alts__": ["--dedup"],
        "help": "Deduplicate identical subterms by introducing a let-binding. This reduces size but may increase runtime slightly.",
    },
}
for k in ARGPARSE_ARGS:
    assert (
        k in DEFAULT_CONFIG.__dict__
    ), f"Key {k} not found in CompilationConfig.__dict__"
