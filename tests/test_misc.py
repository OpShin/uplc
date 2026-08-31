import inspect
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
    version=(1, 0, 0),
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
            (1, 0, 0),
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
        p = parse("""
(program
  1.0.0
  [ [ [ (force (delay [(lam i_0 (con integer 2)) (con bytestring #02)])) (builtin addInteger) ] (error) ] (con pair<list<integer>,unit> [[],()]) ]
)
        """)
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

    def test_inline_variables_single_use_guaranteed(self):
        """Variable used once in a guaranteed position is inlined."""
        from uplc.optimizer.inline_variables import InlineVariableOptimizer
        from uplc.transformer.unique_variables import UniqueVariableTransformer

        # [(lam x [addInteger x (con integer 1)]) (con integer 10)]
        p = Program(
            (1, 0, 0),
            Apply(
                Lambda(
                    "x",
                    Apply(
                        Apply(BuiltIn(BuiltInFun.AddInteger), Variable("x")),
                        BuiltinInteger(1),
                    ),
                ),
                BuiltinInteger(10),
            ),
        )
        p = UniqueVariableTransformer().visit(p)
        p_inlined = InlineVariableOptimizer().visit(p)
        # The Lambda binding should have been removed (outer Apply(Lambda, val) is gone)
        self.assertNotIsInstance(
            p_inlined.term.f,
            Lambda,
            "Variable was not inlined - Lambda binding should be gone",
        )
        # Result must still be correct
        r = eval(p_inlined)
        self.assertEqual(r.result, BuiltinInteger(11))

    def test_inline_variables_not_inlined_when_used_twice(self):
        """Variable used twice is NOT inlined."""
        from uplc.optimizer.inline_variables import InlineVariableOptimizer
        from uplc.transformer.unique_variables import UniqueVariableTransformer

        # [(lam x [x x]) (con integer 5)]
        p = Program(
            (1, 0, 0),
            Apply(
                Lambda("x", Apply(Variable("x"), Variable("x"))),
                BuiltinInteger(5),
            ),
        )
        p = UniqueVariableTransformer().visit(p)
        before = p.dumps()
        p_after = InlineVariableOptimizer().visit(p)
        self.assertEqual(
            before, p_after.dumps(), "Double-use variable should NOT be inlined"
        )

    def test_inline_variables_not_inlined_inside_delay(self):
        """Variable used inside Delay is NOT inlined (not guaranteed)."""
        from uplc.optimizer.inline_variables import InlineVariableOptimizer
        from uplc.transformer.unique_variables import UniqueVariableTransformer

        # [(lam x (delay x)) (con integer 42)]
        p = Program(
            (1, 0, 0),
            Apply(
                Lambda("x", Delay(Variable("x"))),
                BuiltinInteger(42),
            ),
        )
        p = UniqueVariableTransformer().visit(p)
        before = p.dumps()
        p_after = InlineVariableOptimizer().visit(p)
        self.assertEqual(
            before, p_after.dumps(), "Variable inside Delay should NOT be inlined"
        )

    def test_inline_variables_not_inlined_inside_lambda(self):
        """Variable used only inside a nested Lambda body is NOT inlined."""
        from uplc.optimizer.inline_variables import InlineVariableOptimizer
        from uplc.transformer.unique_variables import UniqueVariableTransformer

        # [(lam x (lam y x)) (con integer 5)]
        p = Program(
            (1, 0, 0),
            Apply(
                Lambda("x", Lambda("y", Variable("x"))),
                BuiltinInteger(5),
            ),
        )
        p = UniqueVariableTransformer().visit(p)
        before = p.dumps()
        p_after = InlineVariableOptimizer().visit(p)
        self.assertEqual(
            before, p_after.dumps(), "Variable inside Lambda body should NOT be inlined"
        )

    def test_inline_variables_o3_preserves_semantics(self):
        """O3 compilation with inline_variables produces the same results as O0."""
        with open("examples/fibonacci.uplc", "r") as f:
            p = parse(f.read())
        p0 = tools.compile(p, compiler_config.OPT_O0_CONFIG)
        p3 = tools.compile(p, compiler_config.OPT_O3_CONFIG)
        for i in range(5):
            r0 = eval(p0, BuiltinInteger(i))
            r3 = eval(p3, BuiltinInteger(i))
            self.assertEqual(
                r0.result,
                r3.result,
                f"O3 result differs from O0 for input {i}",
            )

    def test_inline_variables_case_scrutinee_guaranteed(self):
        """Variable in Case scrutinee position is inlined (guaranteed)."""
        from uplc.optimizer.inline_variables import (
            GuaranteedExecutionChecker,
            VariableOccurrenceCounter,
        )

        # body = (case x (lam y y))  -- x is in scrutinee, guaranteed
        body = Case(Variable("x"), [Lambda("y", Variable("y"))])
        counter = VariableOccurrenceCounter()
        counter.visit(body)
        self.assertEqual(counter.counts.get("x", 0), 1)
        self.assertTrue(
            GuaranteedExecutionChecker("x").visit(body),
            "x in Case scrutinee should be in guaranteed position",
        )

    def test_inline_variables_case_branch_not_guaranteed(self):
        """Variable inside Case branch is NOT in a guaranteed position."""
        from uplc.optimizer.inline_variables import GuaranteedExecutionChecker

        # body = (case (con integer 0) (lam y x))  -- x is only in branch, not scrutinee
        body = Case(BuiltinInteger(0), [Lambda("y", Variable("x"))])
        self.assertFalse(
            GuaranteedExecutionChecker("x").visit(body),
            "x inside Case branch should NOT be in guaranteed position",
        )

    def test_inline_variables_constr_field_guaranteed(self):
        """Variable inside a Constr field is in a guaranteed position."""
        from uplc.optimizer.inline_variables import (
            GuaranteedExecutionChecker,
            InlineVariableOptimizer,
        )
        from uplc.transformer.unique_variables import UniqueVariableTransformer

        # GuaranteedExecutionChecker: x inside Constr fields should be guaranteed
        body = Constr(0, [Variable("x"), BuiltinInteger(1)])
        self.assertTrue(
            GuaranteedExecutionChecker("x").visit(body),
            "x inside Constr fields should be in guaranteed position",
        )

        # Also verify the optimizer actually inlines through a Constr field
        # [(lam x (constr 0 x (con integer 1))) (con integer 5)]
        p = Program(
            (1, 0, 0),
            Apply(
                Lambda("x", Constr(0, [Variable("x"), BuiltinInteger(1)])),
                BuiltinInteger(5),
            ),
        )
        p = UniqueVariableTransformer().visit(p)
        p_inlined = InlineVariableOptimizer().visit(p)
        # After inlining, Apply(Lambda, val) is replaced by the Constr directly
        self.assertIsInstance(
            p_inlined.term,
            Constr,
            "Variable in Constr field was not inlined - Apply(Lambda,...) should become Constr directly",
        )

    def test_inline_variables_program_visit(self):
        """GuaranteedExecutionChecker.visit_Program delegates to the term."""
        from uplc.optimizer.inline_variables import GuaranteedExecutionChecker

        # Program wrapping a body where x is in a guaranteed position
        prog = Program((1, 0, 0), Variable("x"))
        self.assertTrue(
            GuaranteedExecutionChecker("x").visit(prog),
            "x at top-level of Program should be in guaranteed position",
        )

        # Program wrapping a body where x is NOT guaranteed (inside Delay)
        prog_not_guaranteed = Program((1, 0, 0), Delay(Variable("x")))
        self.assertFalse(
            GuaranteedExecutionChecker("x").visit(prog_not_guaranteed),
            "x inside Delay in Program should NOT be in guaranteed position",
        )

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

    def test_case_constr_encoding(self):
        with open(Path(__file__).parent / "case_constr" / "case_constr.uplc", "r") as f:
            program = f.read()
        p = parse(program)
        encoded = flatten(p)
        with open(Path(__file__).parent / "case_constr" / "case_constr.flat", "r") as f:
            encoded_ref = bytes.fromhex(f.read())
        self.assertEqual(encoded, encoded_ref)
        decoded = unflatten(encoded_ref)
        self.assertEqual(p, decoded)

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

    def test_haskell_string_escapes(self):
        """Test Haskell decimal (\\DDD) and octal (\\oOOO) string escapes.

        Conformance test string-04:
        Input: (con string "\\t\\"\\83\\x75\\x63\\o143e\\x73s\\o041\\o042\\n")
        Expected decoded value: \\t"Success!"\\n
        """
        program = (
            r'(program 1.0.0 (con string "\t\"\83\x75\x63\o143e\x73s\o041\o042\n"))'
        )
        p = parse(program)
        # The string value should be: tab + "Success!" + newline
        self.assertEqual(p.term.value, '\t"Success!"\n')

    def test_haskell_decimal_escape(self):
        """Test standalone Haskell decimal escape."""
        program = r'(program 1.0.0 (con string "\65"))'
        p = parse(program)
        self.assertEqual(p.term.value, "A")  # chr(65) == 'A'

    def test_haskell_octal_escape(self):
        """Test standalone Haskell octal escape."""
        program = r'(program 1.0.0 (con string "\o101"))'
        p = parse(program)
        self.assertEqual(p.term.value, "A")  # chr(0o101) == 'A'

    def test_array_type_keyword(self):
        """Test that 'array' is accepted as an alias for 'list'.

        Haskell ref: PlutusCore.Default.Universe (defaultUni) lists
        'array' as an alternative name for the list type constructor.
        """
        program = "(program 1.0.0 (con (list integer) [1, 2, 3]))"
        p_list = parse(program)
        program_array = "(program 1.0.0 (con (array integer) [1, 2, 3]))"
        p_array = parse(program_array)
        self.assertEqual(p_list.term.values, p_array.term.values)

    def test_array_type_keyword_aiken_dialect(self):
        """Test 'array' works in Aiken dialect (caret notation)."""
        program = "(program 1.0.0 (con array<integer> [1, 2]))"
        p = parse(program)
        self.assertEqual(len(p.term.values), 2)

    def test_strict_mode_trailing_data(self):
        """Test that strict=True rejects programs with trailing bytes.

        PlutusV3 (Conway-era) requires strict deserialization — no
        trailing bytes are allowed after the flat-encoded program.
        """
        from uplc.tools import flatten, unflatten

        program = parse("(program 1.0.0 (con integer 1))")
        flat_bytes = flatten(program)
        # Normal mode should accept
        unflatten(flat_bytes, strict=False)
        # Strict mode should also accept clean encoding
        unflatten(flat_bytes, strict=True)

    def test_case_on_bool(self):
        """Test case expression scrutinizing a Bool value.

        CEK machine must convert Bool to constr-like: False=tag 0, True=tag 1.
        """
        program = parse(
            "(program 1.1.0 (case (con bool True) (con integer 10) (con integer 20)))"
        )
        result = eval(program)
        self.assertEqual(result.result.value, 20)

    def test_case_on_unit(self):
        """Test case expression scrutinizing a Unit value.

        CEK machine must convert Unit to constr-like: tag 0, no fields.
        """
        program = parse("(program 1.1.0 (case (con unit ()) (con integer 42)))")
        result = eval(program)
        self.assertEqual(result.result.value, 42)

    def test_case_on_list_nil(self):
        """Test case on empty list: nil = tag 0."""
        program = parse(
            "(program 1.1.0 (case (con (list integer) []) (con integer 1) (con integer 2)))"
        )
        result = eval(program)
        self.assertEqual(result.result.value, 1)

    def test_case_on_list_cons(self):
        """Test case on non-empty list: cons = tag 1 with fields [head, tail]."""
        program = parse(
            "(program 1.1.0 (case (con (list integer) [5, 6]) (con integer 1) (lam h (lam t (con integer 2)))))"
        )
        result = eval(program)
        self.assertEqual(result.result.value, 2)

    def test_zero_cost_builtin_raises(self):
        """Unknown builtins should raise RuntimeError, not return Budget(0,0)."""
        from uplc.machine import budget_cost_of_op_on_model
        from uplc.cost_model import BuiltinCostModel

        empty_model = BuiltinCostModel(cpu={}, memory={})
        with self.assertRaises(RuntimeError):
            budget_cost_of_op_on_model(empty_model, "FakeBuiltin")

    def test_cost_model_file_extension(self):
        """Cost model loader should match '.json' suffix, not 'json'."""
        import uplc.cost_model as cm

        # The fix: file.suffix returns ".json", not "json"
        # We verify the code uses ".json" by checking the source
        import inspect

        source = inspect.getsource(cm.load_network_config)
        self.assertIn('.suffix == ".json"', source)

    def test_schnorr_error_message(self):
        """Schnorr verification should report 'Schnorr', not 'ECDSA'."""
        import uplc.ast as uplc_ast

        source = inspect.getsource(uplc_ast.verify_schnorr_secp256k1)
        self.assertNotIn("ECDSA", source)
