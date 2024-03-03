from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class CompilationConfig:
    constant_folding: Optional[bool] = None

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
OPT_O1_CONFIG = OPT_O0_CONFIG.update()
OPT_O2_CONFIG = OPT_O1_CONFIG.update(constant_folding=True)
OPT_O3_CONFIG = OPT_O2_CONFIG.update()
OPT_CONFIGS = [OPT_O0_CONFIG, OPT_O1_CONFIG, OPT_O2_CONFIG, OPT_O3_CONFIG]

DEFAULT_CONFIG = CompilationConfig().update(OPT_O1_CONFIG)

ARGPARSE_ARGS = {
    "unique_variable_names": {
        "__alts__": ["--unique-varnames"],
        "help": "Assign variables a unique name.",
    },
    "constant_folding": {
        "__alts__": ["--cf"],
        "help": "Enables experimental constant folding, including propagation and code execution.",
    },
}
for k in ARGPARSE_ARGS:
    assert (
        k in DEFAULT_CONFIG.__dict__
    ), f"Key {k} not found in CompilationConfig.__dict__"
