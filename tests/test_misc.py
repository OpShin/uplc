import unittest
from pathlib import Path

import rply.parser
from parameterized import parameterized

from uplc import *
from uplc import compiler_config
from uplc.transformer import unique_variables
from uplc.optimizer import pre_evaluation, remove_force_delay
from uplc.lexer import strip_comments
from uplc.ast import *

SAMPLE_CONTRACT = p = Program(
    version=(0, 0, 1),
    term=Apply(
        Apply(
            Apply(
                Lambda(
                    var_name="p0",
                    term=Lambda(
                        var_name="p1",
                        term=Lambda(
                            var_name="p2",
                            term=Force(
                                term=Apply(
                                    f=Apply(
                                        f=Apply(
                                            f=Force(
                                                term=BuiltIn(
                                                    builtin=BuiltInFun.IfThenElse,
                                                    applied_forces=0,
                                                    bound_arguments=[],
                                                )
                                            ),
                                            x=Apply(
                                                f=Lambda(
                                                    var_name="s",
                                                    term=Apply(
                                                        f=Lambda(
                                                            var_name="g",
                                                            term=Apply(
                                                                f=Apply(
                                                                    f=Apply(
                                                                        f=Apply(
                                                                            f=Apply(
                                                                                f=Variable(
                                                                                    name="g"
                                                                                ),
                                                                                x=Variable(
                                                                                    name="g"
                                                                                ),
                                                                            ),
                                                                            x=Apply(
                                                                                f=BuiltIn(
                                                                                    builtin=BuiltInFun.UnIData,
                                                                                    applied_forces=0,
                                                                                    bound_arguments=[],
                                                                                ),
                                                                                x=Variable(
                                                                                    name="p0"
                                                                                ),
                                                                            ),
                                                                        ),
                                                                        x=Apply(
                                                                            f=BuiltIn(
                                                                                builtin=BuiltInFun.UnIData,
                                                                                applied_forces=0,
                                                                                bound_arguments=[],
                                                                            ),
                                                                            x=Variable(
                                                                                name="p1"
                                                                            ),
                                                                        ),
                                                                    ),
                                                                    x=Variable(
                                                                        name="p2"
                                                                    ),
                                                                ),
                                                                x=Variable(name="s"),
                                                            ),
                                                            state=frozendict.frozendict(
                                                                {}
                                                            ),
                                                        ),
                                                        x=Force(
                                                            term=Apply(
                                                                f=Apply(
                                                                    f=Apply(
                                                                        f=Lambda(
                                                                            var_name="s",
                                                                            term=Apply(
                                                                                f=Lambda(
                                                                                    var_name="s",
                                                                                    term=Lambda(
                                                                                        var_name="x",
                                                                                        term=Lambda(
                                                                                            var_name="def",
                                                                                            term=Force(
                                                                                                term=Apply(
                                                                                                    f=Apply(
                                                                                                        f=Apply(
                                                                                                            f=Force(
                                                                                                                term=BuiltIn(
                                                                                                                    builtin=BuiltInFun.IfThenElse,
                                                                                                                    applied_forces=0,
                                                                                                                    bound_arguments=[],
                                                                                                                )
                                                                                                            ),
                                                                                                            x=Apply(
                                                                                                                f=Apply(
                                                                                                                    f=BuiltIn(
                                                                                                                        builtin=BuiltInFun.EqualsByteString,
                                                                                                                        applied_forces=0,
                                                                                                                        bound_arguments=[],
                                                                                                                    ),
                                                                                                                    x=Variable(
                                                                                                                        name="x"
                                                                                                                    ),
                                                                                                                ),
                                                                                                                x=BuiltinByteString(
                                                                                                                    value=b"validator"
                                                                                                                ),
                                                                                                            ),
                                                                                                        ),
                                                                                                        x=Delay(
                                                                                                            term=Delay(
                                                                                                                term=Lambda(
                                                                                                                    var_name="f",
                                                                                                                    term=Lambda(
                                                                                                                        var_name="p0",
                                                                                                                        term=Lambda(
                                                                                                                            var_name="p1",
                                                                                                                            term=Lambda(
                                                                                                                                var_name="p2",
                                                                                                                                term=Lambda(
                                                                                                                                    var_name="s",
                                                                                                                                    term=Apply(
                                                                                                                                        f=Lambda(
                                                                                                                                            var_name="s",
                                                                                                                                            term=Apply(
                                                                                                                                                f=Apply(
                                                                                                                                                    f=BuiltIn(
                                                                                                                                                        builtin=BuiltInFun.EqualsInteger,
                                                                                                                                                        applied_forces=0,
                                                                                                                                                        bound_arguments=[],
                                                                                                                                                    ),
                                                                                                                                                    x=Apply(
                                                                                                                                                        f=Lambda(
                                                                                                                                                            var_name="s",
                                                                                                                                                            term=Apply(
                                                                                                                                                                f=Apply(
                                                                                                                                                                    f=BuiltIn(
                                                                                                                                                                        builtin=BuiltInFun.AddInteger,
                                                                                                                                                                        applied_forces=0,
                                                                                                                                                                        bound_arguments=[],
                                                                                                                                                                    ),
                                                                                                                                                                    x=Apply(
                                                                                                                                                                        f=Lambda(
                                                                                                                                                                            var_name="s",
                                                                                                                                                                            term=Force(
                                                                                                                                                                                term=Apply(
                                                                                                                                                                                    f=Apply(
                                                                                                                                                                                        f=Variable(
                                                                                                                                                                                            name="s"
                                                                                                                                                                                        ),
                                                                                                                                                                                        x=BuiltinByteString(
                                                                                                                                                                                            value=b"datum"
                                                                                                                                                                                        ),
                                                                                                                                                                                    ),
                                                                                                                                                                                    x=Delay(
                                                                                                                                                                                        term=Apply(
                                                                                                                                                                                            f=Lambda(
                                                                                                                                                                                                var_name="_",
                                                                                                                                                                                                term=Error(),
                                                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                                                    {}
                                                                                                                                                                                                ),
                                                                                                                                                                                            ),
                                                                                                                                                                                            x=Apply(
                                                                                                                                                                                                f=Apply(
                                                                                                                                                                                                    f=Force(
                                                                                                                                                                                                        term=BuiltIn(
                                                                                                                                                                                                            builtin=BuiltInFun.Trace,
                                                                                                                                                                                                            applied_forces=0,
                                                                                                                                                                                                            bound_arguments=[],
                                                                                                                                                                                                        )
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    x=BuiltinString(
                                                                                                                                                                                                        value="NameError"
                                                                                                                                                                                                    ),
                                                                                                                                                                                                ),
                                                                                                                                                                                                x=BuiltinUnit(),
                                                                                                                                                                                            ),
                                                                                                                                                                                        ),
                                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                                            {}
                                                                                                                                                                                        ),
                                                                                                                                                                                    ),
                                                                                                                                                                                )
                                                                                                                                                                            ),
                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                {}
                                                                                                                                                                            ),
                                                                                                                                                                        ),
                                                                                                                                                                        x=Variable(
                                                                                                                                                                            name="s"
                                                                                                                                                                        ),
                                                                                                                                                                    ),
                                                                                                                                                                ),
                                                                                                                                                                x=Apply(
                                                                                                                                                                    f=Lambda(
                                                                                                                                                                        var_name="s",
                                                                                                                                                                        term=Force(
                                                                                                                                                                            term=Apply(
                                                                                                                                                                                f=Apply(
                                                                                                                                                                                    f=Variable(
                                                                                                                                                                                        name="s"
                                                                                                                                                                                    ),
                                                                                                                                                                                    x=BuiltinByteString(
                                                                                                                                                                                        value=b"redeemer"
                                                                                                                                                                                    ),
                                                                                                                                                                                ),
                                                                                                                                                                                x=Delay(
                                                                                                                                                                                    term=Apply(
                                                                                                                                                                                        f=Lambda(
                                                                                                                                                                                            var_name="_",
                                                                                                                                                                                            term=Error(),
                                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                                {}
                                                                                                                                                                                            ),
                                                                                                                                                                                        ),
                                                                                                                                                                                        x=Apply(
                                                                                                                                                                                            f=Apply(
                                                                                                                                                                                                f=Force(
                                                                                                                                                                                                    term=BuiltIn(
                                                                                                                                                                                                        builtin=BuiltInFun.Trace,
                                                                                                                                                                                                        applied_forces=0,
                                                                                                                                                                                                        bound_arguments=[],
                                                                                                                                                                                                    )
                                                                                                                                                                                                ),
                                                                                                                                                                                                x=BuiltinString(
                                                                                                                                                                                                    value="NameError"
                                                                                                                                                                                                ),
                                                                                                                                                                                            ),
                                                                                                                                                                                            x=BuiltinUnit(),
                                                                                                                                                                                        ),
                                                                                                                                                                                    ),
                                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                                        {}
                                                                                                                                                                                    ),
                                                                                                                                                                                ),
                                                                                                                                                                            )
                                                                                                                                                                        ),
                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                            {}
                                                                                                                                                                        ),
                                                                                                                                                                    ),
                                                                                                                                                                    x=Variable(
                                                                                                                                                                        name="s"
                                                                                                                                                                    ),
                                                                                                                                                                ),
                                                                                                                                                            ),
                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                {}
                                                                                                                                                            ),
                                                                                                                                                        ),
                                                                                                                                                        x=Variable(
                                                                                                                                                            name="s"
                                                                                                                                                        ),
                                                                                                                                                    ),
                                                                                                                                                ),
                                                                                                                                                x=Apply(
                                                                                                                                                    f=Lambda(
                                                                                                                                                        var_name="s",
                                                                                                                                                        term=BuiltinInteger(
                                                                                                                                                            value=42
                                                                                                                                                        ),
                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                            {}
                                                                                                                                                        ),
                                                                                                                                                    ),
                                                                                                                                                    x=Variable(
                                                                                                                                                        name="s"
                                                                                                                                                    ),
                                                                                                                                                ),
                                                                                                                                            ),
                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                {}
                                                                                                                                            ),
                                                                                                                                        ),
                                                                                                                                        x=Apply(
                                                                                                                                            f=Lambda(
                                                                                                                                                var_name="s",
                                                                                                                                                term=Variable(
                                                                                                                                                    name="s"
                                                                                                                                                ),
                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                    {}
                                                                                                                                                ),
                                                                                                                                            ),
                                                                                                                                            x=Lambda(
                                                                                                                                                var_name="x",
                                                                                                                                                term=Lambda(
                                                                                                                                                    var_name="def",
                                                                                                                                                    term=Force(
                                                                                                                                                        term=Apply(
                                                                                                                                                            f=Apply(
                                                                                                                                                                f=Apply(
                                                                                                                                                                    f=Force(
                                                                                                                                                                        term=BuiltIn(
                                                                                                                                                                            builtin=BuiltInFun.IfThenElse,
                                                                                                                                                                            applied_forces=0,
                                                                                                                                                                            bound_arguments=[],
                                                                                                                                                                        )
                                                                                                                                                                    ),
                                                                                                                                                                    x=Apply(
                                                                                                                                                                        f=Apply(
                                                                                                                                                                            f=BuiltIn(
                                                                                                                                                                                builtin=BuiltInFun.EqualsByteString,
                                                                                                                                                                                applied_forces=0,
                                                                                                                                                                                bound_arguments=[],
                                                                                                                                                                            ),
                                                                                                                                                                            x=Variable(
                                                                                                                                                                                name="x"
                                                                                                                                                                            ),
                                                                                                                                                                        ),
                                                                                                                                                                        x=BuiltinByteString(
                                                                                                                                                                            value=b"context"
                                                                                                                                                                        ),
                                                                                                                                                                    ),
                                                                                                                                                                ),
                                                                                                                                                                x=Delay(
                                                                                                                                                                    term=Delay(
                                                                                                                                                                        term=Variable(
                                                                                                                                                                            name="p2"
                                                                                                                                                                        ),
                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                            {}
                                                                                                                                                                        ),
                                                                                                                                                                    ),
                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                        {}
                                                                                                                                                                    ),
                                                                                                                                                                ),
                                                                                                                                                            ),
                                                                                                                                                            x=Delay(
                                                                                                                                                                term=Force(
                                                                                                                                                                    term=Apply(
                                                                                                                                                                        f=Apply(
                                                                                                                                                                            f=Apply(
                                                                                                                                                                                f=Force(
                                                                                                                                                                                    term=BuiltIn(
                                                                                                                                                                                        builtin=BuiltInFun.IfThenElse,
                                                                                                                                                                                        applied_forces=0,
                                                                                                                                                                                        bound_arguments=[],
                                                                                                                                                                                    )
                                                                                                                                                                                ),
                                                                                                                                                                                x=Apply(
                                                                                                                                                                                    f=Apply(
                                                                                                                                                                                        f=BuiltIn(
                                                                                                                                                                                            builtin=BuiltInFun.EqualsByteString,
                                                                                                                                                                                            applied_forces=0,
                                                                                                                                                                                            bound_arguments=[],
                                                                                                                                                                                        ),
                                                                                                                                                                                        x=Variable(
                                                                                                                                                                                            name="x"
                                                                                                                                                                                        ),
                                                                                                                                                                                    ),
                                                                                                                                                                                    x=BuiltinByteString(
                                                                                                                                                                                        value=b"redeemer"
                                                                                                                                                                                    ),
                                                                                                                                                                                ),
                                                                                                                                                                            ),
                                                                                                                                                                            x=Delay(
                                                                                                                                                                                term=Delay(
                                                                                                                                                                                    term=Variable(
                                                                                                                                                                                        name="p1"
                                                                                                                                                                                    ),
                                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                                        {}
                                                                                                                                                                                    ),
                                                                                                                                                                                ),
                                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                                    {}
                                                                                                                                                                                ),
                                                                                                                                                                            ),
                                                                                                                                                                        ),
                                                                                                                                                                        x=Delay(
                                                                                                                                                                            term=Force(
                                                                                                                                                                                term=Apply(
                                                                                                                                                                                    f=Apply(
                                                                                                                                                                                        f=Apply(
                                                                                                                                                                                            f=Force(
                                                                                                                                                                                                term=BuiltIn(
                                                                                                                                                                                                    builtin=BuiltInFun.IfThenElse,
                                                                                                                                                                                                    applied_forces=0,
                                                                                                                                                                                                    bound_arguments=[],
                                                                                                                                                                                                )
                                                                                                                                                                                            ),
                                                                                                                                                                                            x=Apply(
                                                                                                                                                                                                f=Apply(
                                                                                                                                                                                                    f=BuiltIn(
                                                                                                                                                                                                        builtin=BuiltInFun.EqualsByteString,
                                                                                                                                                                                                        applied_forces=0,
                                                                                                                                                                                                        bound_arguments=[],
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    x=Variable(
                                                                                                                                                                                                        name="x"
                                                                                                                                                                                                    ),
                                                                                                                                                                                                ),
                                                                                                                                                                                                x=BuiltinByteString(
                                                                                                                                                                                                    value=b"datum"
                                                                                                                                                                                                ),
                                                                                                                                                                                            ),
                                                                                                                                                                                        ),
                                                                                                                                                                                        x=Delay(
                                                                                                                                                                                            term=Delay(
                                                                                                                                                                                                term=Variable(
                                                                                                                                                                                                    name="p0"
                                                                                                                                                                                                ),
                                                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                                                    {}
                                                                                                                                                                                                ),
                                                                                                                                                                                            ),
                                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                                {}
                                                                                                                                                                                            ),
                                                                                                                                                                                        ),
                                                                                                                                                                                    ),
                                                                                                                                                                                    x=Delay(
                                                                                                                                                                                        term=Force(
                                                                                                                                                                                            term=Apply(
                                                                                                                                                                                                f=Apply(
                                                                                                                                                                                                    f=Apply(
                                                                                                                                                                                                        f=Force(
                                                                                                                                                                                                            term=BuiltIn(
                                                                                                                                                                                                                builtin=BuiltInFun.IfThenElse,
                                                                                                                                                                                                                applied_forces=0,
                                                                                                                                                                                                                bound_arguments=[],
                                                                                                                                                                                                            )
                                                                                                                                                                                                        ),
                                                                                                                                                                                                        x=Apply(
                                                                                                                                                                                                            f=Apply(
                                                                                                                                                                                                                f=BuiltIn(
                                                                                                                                                                                                                    builtin=BuiltInFun.EqualsByteString,
                                                                                                                                                                                                                    applied_forces=0,
                                                                                                                                                                                                                    bound_arguments=[],
                                                                                                                                                                                                                ),
                                                                                                                                                                                                                x=Variable(
                                                                                                                                                                                                                    name="x"
                                                                                                                                                                                                                ),
                                                                                                                                                                                                            ),
                                                                                                                                                                                                            x=BuiltinByteString(
                                                                                                                                                                                                                value=b"validator"
                                                                                                                                                                                                            ),
                                                                                                                                                                                                        ),
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    x=Delay(
                                                                                                                                                                                                        term=Delay(
                                                                                                                                                                                                            term=Variable(
                                                                                                                                                                                                                name="f"
                                                                                                                                                                                                            ),
                                                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                                                {}
                                                                                                                                                                                                            ),
                                                                                                                                                                                                        ),
                                                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                                                            {}
                                                                                                                                                                                                        ),
                                                                                                                                                                                                    ),
                                                                                                                                                                                                ),
                                                                                                                                                                                                x=Delay(
                                                                                                                                                                                                    term=Apply(
                                                                                                                                                                                                        f=Apply(
                                                                                                                                                                                                            f=Variable(
                                                                                                                                                                                                                name="s"
                                                                                                                                                                                                            ),
                                                                                                                                                                                                            x=Variable(
                                                                                                                                                                                                                name="x"
                                                                                                                                                                                                            ),
                                                                                                                                                                                                        ),
                                                                                                                                                                                                        x=Variable(
                                                                                                                                                                                                            name="def"
                                                                                                                                                                                                        ),
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                                                        {}
                                                                                                                                                                                                    ),
                                                                                                                                                                                                ),
                                                                                                                                                                                            )
                                                                                                                                                                                        ),
                                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                                            {}
                                                                                                                                                                                        ),
                                                                                                                                                                                    ),
                                                                                                                                                                                )
                                                                                                                                                                            ),
                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                {}
                                                                                                                                                                            ),
                                                                                                                                                                        ),
                                                                                                                                                                    )
                                                                                                                                                                ),
                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                    {}
                                                                                                                                                                ),
                                                                                                                                                            ),
                                                                                                                                                        )
                                                                                                                                                    ),
                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                        {}
                                                                                                                                                    ),
                                                                                                                                                ),
                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                    {}
                                                                                                                                                ),
                                                                                                                                            ),
                                                                                                                                        ),
                                                                                                                                    ),
                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                        {}
                                                                                                                                    ),
                                                                                                                                ),
                                                                                                                                state=frozendict.frozendict(
                                                                                                                                    {}
                                                                                                                                ),
                                                                                                                            ),
                                                                                                                            state=frozendict.frozendict(
                                                                                                                                {}
                                                                                                                            ),
                                                                                                                        ),
                                                                                                                        state=frozendict.frozendict(
                                                                                                                            {}
                                                                                                                        ),
                                                                                                                    ),
                                                                                                                    state=frozendict.frozendict(
                                                                                                                        {}
                                                                                                                    ),
                                                                                                                ),
                                                                                                                state=frozendict.frozendict(
                                                                                                                    {}
                                                                                                                ),
                                                                                                            ),
                                                                                                            state=frozendict.frozendict(
                                                                                                                {}
                                                                                                            ),
                                                                                                        ),
                                                                                                    ),
                                                                                                    x=Delay(
                                                                                                        term=Apply(
                                                                                                            f=Apply(
                                                                                                                f=Variable(
                                                                                                                    name="s"
                                                                                                                ),
                                                                                                                x=Variable(
                                                                                                                    name="x"
                                                                                                                ),
                                                                                                            ),
                                                                                                            x=Variable(
                                                                                                                name="def"
                                                                                                            ),
                                                                                                        ),
                                                                                                        state=frozendict.frozendict(
                                                                                                            {}
                                                                                                        ),
                                                                                                    ),
                                                                                                )
                                                                                            ),
                                                                                            state=frozendict.frozendict(
                                                                                                {}
                                                                                            ),
                                                                                        ),
                                                                                        state=frozendict.frozendict(
                                                                                            {}
                                                                                        ),
                                                                                    ),
                                                                                    state=frozendict.frozendict(
                                                                                        {}
                                                                                    ),
                                                                                ),
                                                                                x=Apply(
                                                                                    f=Lambda(
                                                                                        var_name="s",
                                                                                        term=Variable(
                                                                                            name="s"
                                                                                        ),
                                                                                        state=frozendict.frozendict(
                                                                                            {}
                                                                                        ),
                                                                                    ),
                                                                                    x=Apply(
                                                                                        f=Lambda(
                                                                                            var_name="s",
                                                                                            term=Variable(
                                                                                                name="s"
                                                                                            ),
                                                                                            state=frozendict.frozendict(
                                                                                                {}
                                                                                            ),
                                                                                        ),
                                                                                        x=Apply(
                                                                                            f=Lambda(
                                                                                                var_name="s",
                                                                                                term=Variable(
                                                                                                    name="s"
                                                                                                ),
                                                                                                state=frozendict.frozendict(
                                                                                                    {}
                                                                                                ),
                                                                                            ),
                                                                                            x=Apply(
                                                                                                f=Lambda(
                                                                                                    var_name="s",
                                                                                                    term=Variable(
                                                                                                        name="s"
                                                                                                    ),
                                                                                                    state=frozendict.frozendict(
                                                                                                        {}
                                                                                                    ),
                                                                                                ),
                                                                                                x=Apply(
                                                                                                    f=Lambda(
                                                                                                        var_name="s",
                                                                                                        term=Variable(
                                                                                                            name="s"
                                                                                                        ),
                                                                                                        state=frozendict.frozendict(
                                                                                                            {}
                                                                                                        ),
                                                                                                    ),
                                                                                                    x=Apply(
                                                                                                        f=Lambda(
                                                                                                            var_name="s",
                                                                                                            term=Variable(
                                                                                                                name="s"
                                                                                                            ),
                                                                                                            state=frozendict.frozendict(
                                                                                                                {}
                                                                                                            ),
                                                                                                        ),
                                                                                                        x=Apply(
                                                                                                            f=Lambda(
                                                                                                                var_name="s",
                                                                                                                term=Variable(
                                                                                                                    name="s"
                                                                                                                ),
                                                                                                                state=frozendict.frozendict(
                                                                                                                    {}
                                                                                                                ),
                                                                                                            ),
                                                                                                            x=Apply(
                                                                                                                f=Lambda(
                                                                                                                    var_name="s",
                                                                                                                    term=Variable(
                                                                                                                        name="s"
                                                                                                                    ),
                                                                                                                    state=frozendict.frozendict(
                                                                                                                        {}
                                                                                                                    ),
                                                                                                                ),
                                                                                                                x=Apply(
                                                                                                                    f=Lambda(
                                                                                                                        var_name="s",
                                                                                                                        term=Variable(
                                                                                                                            name="s"
                                                                                                                        ),
                                                                                                                        state=frozendict.frozendict(
                                                                                                                            {}
                                                                                                                        ),
                                                                                                                    ),
                                                                                                                    x=Apply(
                                                                                                                        f=Lambda(
                                                                                                                            var_name="s",
                                                                                                                            term=Variable(
                                                                                                                                name="s"
                                                                                                                            ),
                                                                                                                            state=frozendict.frozendict(
                                                                                                                                {}
                                                                                                                            ),
                                                                                                                        ),
                                                                                                                        x=Apply(
                                                                                                                            f=Lambda(
                                                                                                                                var_name="s",
                                                                                                                                term=Variable(
                                                                                                                                    name="s"
                                                                                                                                ),
                                                                                                                                state=frozendict.frozendict(
                                                                                                                                    {}
                                                                                                                                ),
                                                                                                                            ),
                                                                                                                            x=Apply(
                                                                                                                                f=Lambda(
                                                                                                                                    var_name="s",
                                                                                                                                    term=Variable(
                                                                                                                                        name="s"
                                                                                                                                    ),
                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                        {}
                                                                                                                                    ),
                                                                                                                                ),
                                                                                                                                x=Apply(
                                                                                                                                    f=Lambda(
                                                                                                                                        var_name="s",
                                                                                                                                        term=Variable(
                                                                                                                                            name="s"
                                                                                                                                        ),
                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                            {}
                                                                                                                                        ),
                                                                                                                                    ),
                                                                                                                                    x=Apply(
                                                                                                                                        f=Lambda(
                                                                                                                                            var_name="s",
                                                                                                                                            term=Variable(
                                                                                                                                                name="s"
                                                                                                                                            ),
                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                {}
                                                                                                                                            ),
                                                                                                                                        ),
                                                                                                                                        x=Apply(
                                                                                                                                            f=Lambda(
                                                                                                                                                var_name="s",
                                                                                                                                                term=Variable(
                                                                                                                                                    name="s"
                                                                                                                                                ),
                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                    {}
                                                                                                                                                ),
                                                                                                                                            ),
                                                                                                                                            x=Apply(
                                                                                                                                                f=Lambda(
                                                                                                                                                    var_name="s",
                                                                                                                                                    term=Variable(
                                                                                                                                                        name="s"
                                                                                                                                                    ),
                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                        {}
                                                                                                                                                    ),
                                                                                                                                                ),
                                                                                                                                                x=Apply(
                                                                                                                                                    f=Lambda(
                                                                                                                                                        var_name="s",
                                                                                                                                                        term=Variable(
                                                                                                                                                            name="s"
                                                                                                                                                        ),
                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                            {}
                                                                                                                                                        ),
                                                                                                                                                    ),
                                                                                                                                                    x=Apply(
                                                                                                                                                        f=Lambda(
                                                                                                                                                            var_name="s",
                                                                                                                                                            term=Variable(
                                                                                                                                                                name="s"
                                                                                                                                                            ),
                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                {}
                                                                                                                                                            ),
                                                                                                                                                        ),
                                                                                                                                                        x=Apply(
                                                                                                                                                            f=Lambda(
                                                                                                                                                                var_name="s",
                                                                                                                                                                term=Variable(
                                                                                                                                                                    name="s"
                                                                                                                                                                ),
                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                    {}
                                                                                                                                                                ),
                                                                                                                                                            ),
                                                                                                                                                            x=Apply(
                                                                                                                                                                f=Lambda(
                                                                                                                                                                    var_name="s",
                                                                                                                                                                    term=Variable(
                                                                                                                                                                        name="s"
                                                                                                                                                                    ),
                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                        {}
                                                                                                                                                                    ),
                                                                                                                                                                ),
                                                                                                                                                                x=Apply(
                                                                                                                                                                    f=Lambda(
                                                                                                                                                                        var_name="s",
                                                                                                                                                                        term=Variable(
                                                                                                                                                                            name="s"
                                                                                                                                                                        ),
                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                            {}
                                                                                                                                                                        ),
                                                                                                                                                                    ),
                                                                                                                                                                    x=Apply(
                                                                                                                                                                        f=Lambda(
                                                                                                                                                                            var_name="s",
                                                                                                                                                                            term=Variable(
                                                                                                                                                                                name="s"
                                                                                                                                                                            ),
                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                {}
                                                                                                                                                                            ),
                                                                                                                                                                        ),
                                                                                                                                                                        x=Apply(
                                                                                                                                                                            f=Lambda(
                                                                                                                                                                                var_name="s",
                                                                                                                                                                                term=Variable(
                                                                                                                                                                                    name="s"
                                                                                                                                                                                ),
                                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                                    {}
                                                                                                                                                                                ),
                                                                                                                                                                            ),
                                                                                                                                                                            x=Apply(
                                                                                                                                                                                f=Lambda(
                                                                                                                                                                                    var_name="s",
                                                                                                                                                                                    term=Variable(
                                                                                                                                                                                        name="s"
                                                                                                                                                                                    ),
                                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                                        {}
                                                                                                                                                                                    ),
                                                                                                                                                                                ),
                                                                                                                                                                                x=Apply(
                                                                                                                                                                                    f=Lambda(
                                                                                                                                                                                        var_name="s",
                                                                                                                                                                                        term=Variable(
                                                                                                                                                                                            name="s"
                                                                                                                                                                                        ),
                                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                                            {}
                                                                                                                                                                                        ),
                                                                                                                                                                                    ),
                                                                                                                                                                                    x=Apply(
                                                                                                                                                                                        f=Lambda(
                                                                                                                                                                                            var_name="s",
                                                                                                                                                                                            term=Variable(
                                                                                                                                                                                                name="s"
                                                                                                                                                                                            ),
                                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                                {}
                                                                                                                                                                                            ),
                                                                                                                                                                                        ),
                                                                                                                                                                                        x=Apply(
                                                                                                                                                                                            f=Lambda(
                                                                                                                                                                                                var_name="s",
                                                                                                                                                                                                term=Variable(
                                                                                                                                                                                                    name="s"
                                                                                                                                                                                                ),
                                                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                                                    {}
                                                                                                                                                                                                ),
                                                                                                                                                                                            ),
                                                                                                                                                                                            x=Apply(
                                                                                                                                                                                                f=Lambda(
                                                                                                                                                                                                    var_name="s",
                                                                                                                                                                                                    term=Variable(
                                                                                                                                                                                                        name="s"
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                                                        {}
                                                                                                                                                                                                    ),
                                                                                                                                                                                                ),
                                                                                                                                                                                                x=Apply(
                                                                                                                                                                                                    f=Lambda(
                                                                                                                                                                                                        var_name="s",
                                                                                                                                                                                                        term=Variable(
                                                                                                                                                                                                            name="s"
                                                                                                                                                                                                        ),
                                                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                                                            {}
                                                                                                                                                                                                        ),
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    x=Apply(
                                                                                                                                                                                                        f=Lambda(
                                                                                                                                                                                                            var_name="s",
                                                                                                                                                                                                            term=Variable(
                                                                                                                                                                                                                name="s"
                                                                                                                                                                                                            ),
                                                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                                                {}
                                                                                                                                                                                                            ),
                                                                                                                                                                                                        ),
                                                                                                                                                                                                        x=Apply(
                                                                                                                                                                                                            f=Lambda(
                                                                                                                                                                                                                var_name="s",
                                                                                                                                                                                                                term=Variable(
                                                                                                                                                                                                                    name="s"
                                                                                                                                                                                                                ),
                                                                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                                                                    {}
                                                                                                                                                                                                                ),
                                                                                                                                                                                                            ),
                                                                                                                                                                                                            x=Apply(
                                                                                                                                                                                                                f=Lambda(
                                                                                                                                                                                                                    var_name="s",
                                                                                                                                                                                                                    term=Variable(
                                                                                                                                                                                                                        name="s"
                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                                                                        {}
                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                ),
                                                                                                                                                                                                                x=Apply(
                                                                                                                                                                                                                    f=Lambda(
                                                                                                                                                                                                                        var_name="s",
                                                                                                                                                                                                                        term=Variable(
                                                                                                                                                                                                                            name="s"
                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                                                                            {}
                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                    x=Apply(
                                                                                                                                                                                                                        f=Lambda(
                                                                                                                                                                                                                            var_name="s",
                                                                                                                                                                                                                            term=Variable(
                                                                                                                                                                                                                                name="s"
                                                                                                                                                                                                                            ),
                                                                                                                                                                                                                            state=frozendict.frozendict(
                                                                                                                                                                                                                                {}
                                                                                                                                                                                                                            ),
                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                        x=Apply(
                                                                                                                                                                                                                            f=Lambda(
                                                                                                                                                                                                                                var_name="s",
                                                                                                                                                                                                                                term=Variable(
                                                                                                                                                                                                                                    name="s"
                                                                                                                                                                                                                                ),
                                                                                                                                                                                                                                state=frozendict.frozendict(
                                                                                                                                                                                                                                    {}
                                                                                                                                                                                                                                ),
                                                                                                                                                                                                                            ),
                                                                                                                                                                                                                            x=Apply(
                                                                                                                                                                                                                                f=Lambda(
                                                                                                                                                                                                                                    var_name="s",
                                                                                                                                                                                                                                    term=Variable(
                                                                                                                                                                                                                                        name="s"
                                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                                                                                                                        {}
                                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                                ),
                                                                                                                                                                                                                                x=Apply(
                                                                                                                                                                                                                                    f=Lambda(
                                                                                                                                                                                                                                        var_name="s",
                                                                                                                                                                                                                                        term=Variable(
                                                                                                                                                                                                                                            name="s"
                                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                                                                                                                            {}
                                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                                    x=Variable(
                                                                                                                                                                                                                                        name="s"
                                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                                ),
                                                                                                                                                                                                                            ),
                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                ),
                                                                                                                                                                                                            ),
                                                                                                                                                                                                        ),
                                                                                                                                                                                                    ),
                                                                                                                                                                                                ),
                                                                                                                                                                                            ),
                                                                                                                                                                                        ),
                                                                                                                                                                                    ),
                                                                                                                                                                                ),
                                                                                                                                                                            ),
                                                                                                                                                                        ),
                                                                                                                                                                    ),
                                                                                                                                                                ),
                                                                                                                                                            ),
                                                                                                                                                        ),
                                                                                                                                                    ),
                                                                                                                                                ),
                                                                                                                                            ),
                                                                                                                                        ),
                                                                                                                                    ),
                                                                                                                                ),
                                                                                                                            ),
                                                                                                                        ),
                                                                                                                    ),
                                                                                                                ),
                                                                                                            ),
                                                                                                        ),
                                                                                                    ),
                                                                                                ),
                                                                                            ),
                                                                                        ),
                                                                                    ),
                                                                                ),
                                                                            ),
                                                                            state=frozendict.frozendict(
                                                                                {}
                                                                            ),
                                                                        ),
                                                                        x=Variable(
                                                                            name="s"
                                                                        ),
                                                                    ),
                                                                    x=BuiltinByteString(
                                                                        value=b"validator"
                                                                    ),
                                                                ),
                                                                x=Delay(
                                                                    term=Apply(
                                                                        f=Lambda(
                                                                            var_name="_",
                                                                            term=Error(),
                                                                            state=frozendict.frozendict(
                                                                                {}
                                                                            ),
                                                                        ),
                                                                        x=Apply(
                                                                            f=Apply(
                                                                                f=Force(
                                                                                    term=BuiltIn(
                                                                                        builtin=BuiltInFun.Trace,
                                                                                        applied_forces=0,
                                                                                        bound_arguments=[],
                                                                                    )
                                                                                ),
                                                                                x=BuiltinString(
                                                                                    value="NameError"
                                                                                ),
                                                                            ),
                                                                            x=BuiltinUnit(),
                                                                        ),
                                                                    ),
                                                                    state=frozendict.frozendict(
                                                                        {}
                                                                    ),
                                                                ),
                                                            )
                                                        ),
                                                    ),
                                                    state=frozendict.frozendict({}),
                                                ),
                                                x=Lambda(
                                                    var_name="x",
                                                    term=Lambda(
                                                        var_name="def",
                                                        term=Force(
                                                            term=Apply(
                                                                f=Apply(
                                                                    f=Apply(
                                                                        f=Force(
                                                                            term=BuiltIn(
                                                                                builtin=BuiltInFun.IfThenElse,
                                                                                applied_forces=0,
                                                                                bound_arguments=[],
                                                                            )
                                                                        ),
                                                                        x=Apply(
                                                                            f=Apply(
                                                                                f=BuiltIn(
                                                                                    builtin=BuiltInFun.EqualsByteString,
                                                                                    applied_forces=0,
                                                                                    bound_arguments=[],
                                                                                ),
                                                                                x=Variable(
                                                                                    name="x"
                                                                                ),
                                                                            ),
                                                                            x=BuiltinByteString(
                                                                                value=b"range"
                                                                            ),
                                                                        ),
                                                                    ),
                                                                    x=Delay(
                                                                        term=Delay(
                                                                            term=Lambda(
                                                                                var_name="f",
                                                                                term=Lambda(
                                                                                    var_name="limit",
                                                                                    term=Lambda(
                                                                                        var_name="s",
                                                                                        term=Apply(
                                                                                            f=Apply(
                                                                                                f=Apply(
                                                                                                    f=Lambda(
                                                                                                        var_name="limit",
                                                                                                        term=Lambda(
                                                                                                            var_name="step",
                                                                                                            term=Apply(
                                                                                                                f=Lambda(
                                                                                                                    var_name="g",
                                                                                                                    term=Apply(
                                                                                                                        f=Variable(
                                                                                                                            name="g"
                                                                                                                        ),
                                                                                                                        x=Variable(
                                                                                                                            name="g"
                                                                                                                        ),
                                                                                                                    ),
                                                                                                                    state=frozendict.frozendict(
                                                                                                                        {}
                                                                                                                    ),
                                                                                                                ),
                                                                                                                x=Lambda(
                                                                                                                    var_name="f",
                                                                                                                    term=Lambda(
                                                                                                                        var_name="cur",
                                                                                                                        term=Force(
                                                                                                                            term=Apply(
                                                                                                                                f=Apply(
                                                                                                                                    f=Apply(
                                                                                                                                        f=Force(
                                                                                                                                            term=BuiltIn(
                                                                                                                                                builtin=BuiltInFun.IfThenElse,
                                                                                                                                                applied_forces=0,
                                                                                                                                                bound_arguments=[],
                                                                                                                                            )
                                                                                                                                        ),
                                                                                                                                        x=Apply(
                                                                                                                                            f=Apply(
                                                                                                                                                f=BuiltIn(
                                                                                                                                                    builtin=BuiltInFun.LessThanInteger,
                                                                                                                                                    applied_forces=0,
                                                                                                                                                    bound_arguments=[],
                                                                                                                                                ),
                                                                                                                                                x=Variable(
                                                                                                                                                    name="cur"
                                                                                                                                                ),
                                                                                                                                            ),
                                                                                                                                            x=Variable(
                                                                                                                                                name="limit"
                                                                                                                                            ),
                                                                                                                                        ),
                                                                                                                                    ),
                                                                                                                                    x=Delay(
                                                                                                                                        term=Apply(
                                                                                                                                            f=Apply(
                                                                                                                                                f=Force(
                                                                                                                                                    term=BuiltIn(
                                                                                                                                                        builtin=BuiltInFun.MkCons,
                                                                                                                                                        applied_forces=0,
                                                                                                                                                        bound_arguments=[],
                                                                                                                                                    )
                                                                                                                                                ),
                                                                                                                                                x=Variable(
                                                                                                                                                    name="cur"
                                                                                                                                                ),
                                                                                                                                            ),
                                                                                                                                            x=Apply(
                                                                                                                                                f=Apply(
                                                                                                                                                    f=Variable(
                                                                                                                                                        name="f"
                                                                                                                                                    ),
                                                                                                                                                    x=Variable(
                                                                                                                                                        name="f"
                                                                                                                                                    ),
                                                                                                                                                ),
                                                                                                                                                x=Apply(
                                                                                                                                                    f=Apply(
                                                                                                                                                        f=BuiltIn(
                                                                                                                                                            builtin=BuiltInFun.AddInteger,
                                                                                                                                                            applied_forces=0,
                                                                                                                                                            bound_arguments=[],
                                                                                                                                                        ),
                                                                                                                                                        x=Variable(
                                                                                                                                                            name="cur"
                                                                                                                                                        ),
                                                                                                                                                    ),
                                                                                                                                                    x=Variable(
                                                                                                                                                        name="step"
                                                                                                                                                    ),
                                                                                                                                                ),
                                                                                                                                            ),
                                                                                                                                        ),
                                                                                                                                        state=frozendict.frozendict(
                                                                                                                                            {}
                                                                                                                                        ),
                                                                                                                                    ),
                                                                                                                                ),
                                                                                                                                x=Delay(
                                                                                                                                    term=Apply(
                                                                                                                                        f=BuiltIn(
                                                                                                                                            builtin=BuiltInFun.MkNilData,
                                                                                                                                            applied_forces=0,
                                                                                                                                            bound_arguments=[],
                                                                                                                                        ),
                                                                                                                                        x=BuiltinUnit(),
                                                                                                                                    ),
                                                                                                                                    state=frozendict.frozendict(
                                                                                                                                        {}
                                                                                                                                    ),
                                                                                                                                ),
                                                                                                                            )
                                                                                                                        ),
                                                                                                                        state=frozendict.frozendict(
                                                                                                                            {}
                                                                                                                        ),
                                                                                                                    ),
                                                                                                                    state=frozendict.frozendict(
                                                                                                                        {}
                                                                                                                    ),
                                                                                                                ),
                                                                                                            ),
                                                                                                            state=frozendict.frozendict(
                                                                                                                {}
                                                                                                            ),
                                                                                                        ),
                                                                                                        state=frozendict.frozendict(
                                                                                                            {}
                                                                                                        ),
                                                                                                    ),
                                                                                                    x=Variable(
                                                                                                        name="limit"
                                                                                                    ),
                                                                                                ),
                                                                                                x=BuiltinInteger(
                                                                                                    value=1
                                                                                                ),
                                                                                            ),
                                                                                            x=BuiltinInteger(
                                                                                                value=0
                                                                                            ),
                                                                                        ),
                                                                                        state=frozendict.frozendict(
                                                                                            {}
                                                                                        ),
                                                                                    ),
                                                                                    state=frozendict.frozendict(
                                                                                        {}
                                                                                    ),
                                                                                ),
                                                                                state=frozendict.frozendict(
                                                                                    {}
                                                                                ),
                                                                            ),
                                                                            state=frozendict.frozendict(
                                                                                {}
                                                                            ),
                                                                        ),
                                                                        state=frozendict.frozendict(
                                                                            {}
                                                                        ),
                                                                    ),
                                                                ),
                                                                x=Delay(
                                                                    term=Force(
                                                                        term=Apply(
                                                                            f=Apply(
                                                                                f=Apply(
                                                                                    f=Force(
                                                                                        term=BuiltIn(
                                                                                            builtin=BuiltInFun.IfThenElse,
                                                                                            applied_forces=0,
                                                                                            bound_arguments=[],
                                                                                        )
                                                                                    ),
                                                                                    x=Apply(
                                                                                        f=Apply(
                                                                                            f=BuiltIn(
                                                                                                builtin=BuiltInFun.EqualsByteString,
                                                                                                applied_forces=0,
                                                                                                bound_arguments=[],
                                                                                            ),
                                                                                            x=Variable(
                                                                                                name="x"
                                                                                            ),
                                                                                        ),
                                                                                        x=BuiltinByteString(
                                                                                            value=b"print"
                                                                                        ),
                                                                                    ),
                                                                                ),
                                                                                x=Delay(
                                                                                    term=Delay(
                                                                                        term=Lambda(
                                                                                            var_name="f",
                                                                                            term=Lambda(
                                                                                                var_name="x",
                                                                                                term=Lambda(
                                                                                                    var_name="s",
                                                                                                    term=Apply(
                                                                                                        f=Apply(
                                                                                                            f=Force(
                                                                                                                term=BuiltIn(
                                                                                                                    builtin=BuiltInFun.Trace,
                                                                                                                    applied_forces=0,
                                                                                                                    bound_arguments=[],
                                                                                                                )
                                                                                                            ),
                                                                                                            x=Variable(
                                                                                                                name="x"
                                                                                                            ),
                                                                                                        ),
                                                                                                        x=Apply(
                                                                                                            f=Apply(
                                                                                                                f=BuiltIn(
                                                                                                                    builtin=BuiltInFun.ConstrData,
                                                                                                                    applied_forces=0,
                                                                                                                    bound_arguments=[],
                                                                                                                ),
                                                                                                                x=BuiltinInteger(
                                                                                                                    value=0
                                                                                                                ),
                                                                                                            ),
                                                                                                            x=Apply(
                                                                                                                f=BuiltIn(
                                                                                                                    builtin=BuiltInFun.MkNilData,
                                                                                                                    applied_forces=0,
                                                                                                                    bound_arguments=[],
                                                                                                                ),
                                                                                                                x=BuiltinUnit(),
                                                                                                            ),
                                                                                                        ),
                                                                                                    ),
                                                                                                    state=frozendict.frozendict(
                                                                                                        {}
                                                                                                    ),
                                                                                                ),
                                                                                                state=frozendict.frozendict(
                                                                                                    {}
                                                                                                ),
                                                                                            ),
                                                                                            state=frozendict.frozendict(
                                                                                                {}
                                                                                            ),
                                                                                        ),
                                                                                        state=frozendict.frozendict(
                                                                                            {}
                                                                                        ),
                                                                                    ),
                                                                                    state=frozendict.frozendict(
                                                                                        {}
                                                                                    ),
                                                                                ),
                                                                            ),
                                                                            x=Delay(
                                                                                term=Apply(
                                                                                    f=Apply(
                                                                                        f=Lambda(
                                                                                            var_name="x",
                                                                                            term=Lambda(
                                                                                                var_name="def",
                                                                                                term=Variable(
                                                                                                    name="def"
                                                                                                ),
                                                                                                state=frozendict.frozendict(
                                                                                                    {}
                                                                                                ),
                                                                                            ),
                                                                                            state=frozendict.frozendict(
                                                                                                {}
                                                                                            ),
                                                                                        ),
                                                                                        x=Variable(
                                                                                            name="x"
                                                                                        ),
                                                                                    ),
                                                                                    x=Variable(
                                                                                        name="def"
                                                                                    ),
                                                                                ),
                                                                                state=frozendict.frozendict(
                                                                                    {}
                                                                                ),
                                                                            ),
                                                                        )
                                                                    ),
                                                                    state=frozendict.frozendict(
                                                                        {}
                                                                    ),
                                                                ),
                                                            )
                                                        ),
                                                        state=frozendict.frozendict({}),
                                                    ),
                                                    state=frozendict.frozendict({}),
                                                ),
                                            ),
                                        ),
                                        x=Delay(
                                            term=BuiltinUnit(),
                                            state=frozendict.frozendict({}),
                                        ),
                                    ),
                                    x=Delay(
                                        term=Apply(
                                            f=Lambda(
                                                var_name="_",
                                                term=Error(),
                                                state=frozendict.frozendict({}),
                                            ),
                                            x=Apply(
                                                f=Apply(
                                                    f=Force(
                                                        term=BuiltIn(
                                                            builtin=BuiltInFun.Trace,
                                                            applied_forces=0,
                                                            bound_arguments=[],
                                                        )
                                                    ),
                                                    x=BuiltinString(
                                                        value="ValidationError"
                                                    ),
                                                ),
                                                x=BuiltinUnit(),
                                            ),
                                        ),
                                        state=frozendict.frozendict({}),
                                    ),
                                )
                            ),
                            state=frozendict.frozendict({}),
                        ),
                        state=frozendict.frozendict({}),
                    ),
                    state=frozendict.frozendict({}),
                ),
                PlutusInteger(20),
            ),
            PlutusInteger(22),
        ),
        BuiltinUnit(),
    ),
)


class MiscTest(unittest.TestCase):
    def test_simple_contract(self):
        p = SAMPLE_CONTRACT
        # should not raise
        d = dumps(p)
        # should not raise
        parse(d)
        # should not raise
        r = eval(p)
        self.assertEqual(r.result, BuiltinUnit())

    def test_unpack_plutus_data(self):
        p = Program(
            (0, 0, 1),
            Apply(
                BuiltIn(BuiltInFun.UnConstrData),
                data_from_cbor(
                    bytes.fromhex(
                        "d8799fd8799fd8799fd8799f581ce3a0254c00994f731550f81239f12a60c9fd3ce9b9b191543152ec22ffd8799fd8799fd8799f581cb1bec305ddc80189dac8b628ee0adfbe5245c53b84e678ed7ec23d75ffffffff581ce3a0254c00994f731550f81239f12a60c9fd3ce9b9b191543152ec221b0000018bcfe56800d8799fd8799f4040ffd8799f581cdda5fdb1002f7389b33e036b6afee82a8189becb6cba852e8b79b4fb480014df1047454e53ffffffd8799fd87a801a00989680ffff"
                    )
                ),
            ),
        )
        # should not raise anything
        d = dumps(p)
        # should not raise anything
        parse(d)
        r = eval(p)
        # should not raise anything
        r.result.dumps()
        self.assertEqual(
            r.result,
            BuiltinPair(
                l_value=BuiltinInteger(value=0),
                r_value=BuiltinList(
                    values=[
                        PlutusConstr(
                            constructor=0,
                            fields=[
                                PlutusConstr(
                                    constructor=0,
                                    fields=[
                                        PlutusConstr(
                                            constructor=0,
                                            fields=[
                                                PlutusByteString(
                                                    value=b'\xe3\xa0%L\x00\x99Os\x15P\xf8\x129\xf1*`\xc9\xfd<\xe9\xb9\xb1\x91T1R\xec"'
                                                )
                                            ],
                                        ),
                                        PlutusConstr(
                                            constructor=0,
                                            fields=[
                                                PlutusConstr(
                                                    constructor=0,
                                                    fields=[
                                                        PlutusConstr(
                                                            constructor=0,
                                                            fields=[
                                                                PlutusByteString(
                                                                    value=b"\xb1\xbe\xc3\x05\xdd\xc8\x01\x89\xda\xc8\xb6(\xee\n\xdf\xbeRE\xc5;\x84\xe6x\xed~\xc2=u"
                                                                )
                                                            ],
                                                        )
                                                    ],
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                                PlutusByteString(
                                    value=b'\xe3\xa0%L\x00\x99Os\x15P\xf8\x129\xf1*`\xc9\xfd<\xe9\xb9\xb1\x91T1R\xec"'
                                ),
                                PlutusInteger(value=1700000000000),
                                PlutusConstr(
                                    constructor=0,
                                    fields=[
                                        PlutusConstr(
                                            constructor=0,
                                            fields=[
                                                PlutusByteString(value=b""),
                                                PlutusByteString(value=b""),
                                            ],
                                        ),
                                        PlutusConstr(
                                            constructor=0,
                                            fields=[
                                                PlutusByteString(
                                                    value=b"\xdd\xa5\xfd\xb1\x00/s\x89\xb3>\x03kj\xfe\xe8*\x81\x89\xbe\xcbl\xba\x85.\x8by\xb4\xfb"
                                                ),
                                                PlutusByteString(
                                                    value=b"\x00\x14\xdf\x10GENS"
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        PlutusConstr(
                            constructor=0,
                            fields=[
                                PlutusConstr(constructor=1, fields=[]),
                                PlutusInteger(value=10000000),
                            ],
                        ),
                    ]
                ),
            ),
        )

    def test_parse(self):
        p = parse(
            """
(program
  1.0.0
  [ [ [ (force (delay [(lam i_0 (con integer 2)) (con bytestring #02)])) (builtin addInteger) ] (error) ] (con pair<list<integer>,unit> [[],()]) ]
)
        """
        )
        print(dumps(p))

    @parameterized.expand(
        [
            """(con (pair (pair (list integer) bytestring) data) (([1], #01), Constr 1 [I 2, B #00, List [I 1, I 2], Map [(I 1, B #01)]]))""",
            """(con data (Map []))""",
            """(con (list data) [I 0, B #])""",
            """(con (pair data data) (I 0, B #))""",
        ]
    )
    def test_parse_constants(self, program):
        program = f"(program 1.0.0 {program})"
        p = parse(program)
        self.assertEqual(dumps(p, UPLCDialect.Plutus), program)

    @parameterized.expand(
        [
            """(con (pair (pair (list integer) bytestring) data) (([1], #01), (Constr 1 [I 2, B #00, List [I 1, I 2], Map [(I 1, B #01)]])))""",
            """(con (list data) [(I 0), (B #)])""",
            """(con (pair data data) ((I 0), (B #)))""",
        ]
    )
    @unittest.expectedFailure
    def test_reject_constants(self, program):
        program = f"(program 1.0.0 {program})"
        p = parse(program)

    def test_simple_contract_rewrite(self):
        p = SAMPLE_CONTRACT
        # should not raise
        p = unique_variables.UniqueVariableTransformer().visit(p)
        # should not raise
        d = dumps(p)
        # should not raise
        parse(d)
        r = eval(p)
        self.assertEqual(r.result, BuiltinUnit())

    @parameterized.expand(
        [
            (
                "-- here is a comment \n and here is normal text",
                "\n and here is normal text",
            ),
            ("-- here is a comment ", ""),
        ]
    )
    def test_strip_comments(self, input, expected):
        self.assertEqual(strip_comments(input), expected)

    @parameterized.expand(
        [
            ("""(program 1.0.0 (con data 123)""",),
            ("""(program 1.0.0 (con data { ByteString #00})""",),
        ]
    )
    def test_parse_fail(self, i):
        try:
            parse(i)
            self.fail("Unexpextedly passed")
        except SyntaxError:
            pass

    def test_simple_contract_optimize(self):
        p = SAMPLE_CONTRACT
        # should not raise
        p = pre_evaluation.PreEvaluationOptimizer().visit(p)
        # should not raise
        d = dumps(p)
        # should not raise
        parse(d)
        r = eval(p)
        self.assertEqual(r.result, BuiltinUnit())

    def test_log_single(self):
        x = "Hello, world!"
        p = Program(
            (1, 0, 0),
            Apply(
                Apply(Force(BuiltIn(BuiltInFun.Trace)), BuiltinString(value=x)),
                BuiltinUnit(),
            ),
        )
        r = eval(p)
        self.assertIn(x, r.logs, "Trace did not produce a log.")
        self.assertEqual(
            r.result, BuiltinUnit(), "Trace did not return second argument"
        )

    def test_log_double(self):
        x = "Hello, world!"
        y = "Hello, world 2!"
        p = Program(
            (1, 0, 0),
            Apply(
                Apply(Force(BuiltIn(BuiltInFun.Trace)), BuiltinString(value=y)),
                Apply(
                    Apply(Force(BuiltIn(BuiltInFun.Trace)), BuiltinString(value=x)),
                    BuiltinUnit(),
                ),
            ),
        )
        r = eval(p)
        self.assertIn(x, r.logs, "Trace did not produce a log for first message.")
        self.assertIn(y, r.logs, "Trace did not produce a log for second message.")
        self.assertEqual(r.logs, [x, y], "Trace did log in correct order.")
        self.assertEqual(
            r.result, BuiltinUnit(), "Trace did not return second argument"
        )

    def test_trace_removal_preeval(self):
        x = "Hello, world!"
        y = "Hello, world 2!"
        p = Program(
            (1, 0, 0),
            Apply(
                Apply(Force(BuiltIn(BuiltInFun.Trace)), BuiltinString(value=y)),
                Apply(
                    Apply(Force(BuiltIn(BuiltInFun.Trace)), BuiltinString(value=x)),
                    BuiltinUnit(),
                ),
            ),
        )
        p = pre_evaluation.PreEvaluationOptimizer(skip_traces=False).visit(p)
        r = eval(p)
        self.assertNotIn(
            x, r.logs, "Trace was produced even though rewrite should have removed it"
        )
        self.assertNotIn(y, r.logs, "Trace did not produce a log for second message.")
        self.assertEqual(r.logs, [], "Trace did log.")
        self.assertEqual(
            r.result, BuiltinUnit(), "Trace did not return second argument"
        )

    def test_no_trace_removal_preeval(self):
        x = "Hello, world!"
        y = "Hello, world 2!"
        p = Program(
            (1, 0, 0),
            Apply(
                Apply(Force(BuiltIn(BuiltInFun.Trace)), BuiltinString(value=y)),
                Apply(
                    Apply(Force(BuiltIn(BuiltInFun.Trace)), BuiltinString(value=x)),
                    BuiltinUnit(),
                ),
            ),
        )
        p = pre_evaluation.PreEvaluationOptimizer(skip_traces=True).visit(p)
        r = eval(p)
        self.assertIn(x, r.logs, "Trace did not produce a log for first message.")
        self.assertIn(y, r.logs, "Trace did not produce a log for second message.")
        self.assertEqual(r.logs, [x, y], "Trace did log in correct order.")
        self.assertEqual(
            r.result, BuiltinUnit(), "Trace did not return second argument"
        )

    def test_force_delay_removal(self):
        p = Program((1, 0, 0), Force(Delay(Error())))
        p = remove_force_delay.ForceDelayRemover().visit(p)
        self.assertEqual(p.term, Error(), "Force-Delay was not removed.")

    def test_compiler_options(self):
        with open("examples/fibonacci.uplc", "r") as f:
            p = parse(f.read())
        p1 = tools.compile(p, compiler_config.OPT_O0_CONFIG)
        p2 = tools.compile(p, compiler_config.OPT_O3_CONFIG)
        self.assertNotEqual(
            p1.dumps(), p2.dumps(), "Compiler options did not change the program."
        )
        for i in range(5):
            r1 = eval(p1, BuiltinInteger(i))
            r2 = eval(p2, BuiltinInteger(i))
            self.assertEqual(
                r1.result,
                r2.result,
                "Compiler options did not produce the same result.",
            )

    def test_append_plutusdata_list(self):
        with open(Path(__file__).parent / "constr.uplc", "r") as f:
            program = f.read()
        p = parse(program)
        r = eval(p)
        self.assertEqual(
            r.result,
            PlutusConstr(0, []),
        )

    def test_invalid_json_combination(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"bytes": 2}'
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("expected bytes in 'bytes'", str(context.exception).lower())

    def test_invalid_json_combination_2(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"int": "0F"}'
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("expected integer in 'int'", str(context.exception).lower())

    def test_invalid_json_combination_3(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"int": 1.2}'
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("expected integer in 'int'", str(context.exception).lower())

    def test_invalid_json_format(self):
        """Test error handling for invalid JSON format"""
        param = '{"int": 42'  # Missing closing brace
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("Invalid JSON", str(context.exception))

    def test_invalid_json_syntax(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"int": }'  # Invalid JSON syntax
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("Invalid JSON", str(context.exception))

    def test_invalid_json_key(self):
        """Test error handling for invalid JSON syntax"""
        param = "{}"  # Invalid JSON syntax
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("Unknown JSON", str(context.exception))

    def test_invalid_constructor_json(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"constructor": "hi", "fields": []}'  # Invalid JSON syntax
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn(
            "expected integer in 'constructor'", str(context.exception).lower()
        )

    def test_invalid_constructor_field_json(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"constructor": 0}'  # Invalid JSON syntax
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("expected key 'fields'", str(context.exception).lower())

    def test_invalid_constructor_fields(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"constructor": 0, "fields": 2}'  # Invalid JSON syntax
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("expected a list", str(context.exception).lower())

    def test_invalid_map(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"map": [{"k": 0}]}'  # Invalid JSON syntax
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("expected a dictionary", str(context.exception).lower())

    def test_invalid_map_2(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"map": {"k": 0}}'  # Invalid JSON syntax
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("expected a list", str(context.exception).lower())

    def test_invalid_list(self):
        """Test error handling for invalid JSON syntax"""
        param = '{"list": 1}'  # Invalid JSON syntax
        with self.assertRaises(ValueError) as context:
            data_from_json(param)
        self.assertIn("expected a list", str(context.exception).lower())
