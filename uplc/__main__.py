import argparse
import enum
import json
import pathlib
import sys

import cbor2
import pycardano

from uplc.compiler_config import (
    DEFAULT_CONFIG,
    OPT_CONFIGS,
    ARGPARSE_ARGS,
    CompilationConfig,
)
from .tools import *
from .ast import Program, Apply
from .transformer import unique_variables
from .cost_model import (
    parse_builtin_cost_model,
    parse_cek_machine_cost_model,
    Budget,
    load_network_config,
    latest_network_config,
    updated_cek_machine_cost_model_from_network_config,
    updated_builtin_cost_model_from_network_config,
)


class Command(enum.Enum):
    eval = "eval"
    parse = "parse"
    build = "build"
    dump = "dump"


def get_args():
    a = argparse.ArgumentParser(
        description="An evaluator and compiler for UPLC written in python."
    )
    a.add_argument(
        "command",
        type=str,
        choices=Command.__members__.keys(),
        help="The command to execute on the input file.",
    )
    a.add_argument(
        "input_file",
        type=str,
        help="The input program to parse (foo.uplc). Set to - for stdin.",
    )
    a.add_argument(
        "-o",
        "--output-directory",
        default="",
        type=str,
        help="The output directory for artefacts of the build command. Defaults to the filename of the compiled contract. of the compiled contract.",
    )
    a.add_argument(
        "--dialect",
        default=UPLCDialect.Plutus.value,
        help="The dialect for dumping the parsed UPLC.",
        choices=[d.value for d in UPLCDialect],
    )
    a.add_argument(
        "--from-cbor",
        action="store_true",
        help="Read hex representation of flattened UPLC, wrapped in cbor.",
    )
    a.add_argument(
        "--from-hex",
        action="store_true",
        help="Read hex representation of flattened UPLC, not wrapped in cbor.",
    )
    a.add_argument(
        "args",
        nargs="*",
        default=[],
        help="Input parameters for the function, in case the command is eval.",
    )
    a.add_argument(
        "--recursion-limit",
        default=sys.getrecursionlimit(),
        help="Modify the recursion limit (necessary for larger UPLC programs)",
        type=int,
    )
    a.add_argument(
        "--builtin-cost-model-file",
        default=None,
        help="Provide a builtin cost-model file for the eval command. You will usually not need this.",
        type=str,
    )
    a.add_argument(
        "--cek-machine-cost-model-file",
        default=None,
        help="Provide a builtin cost-model file for the eval command. You will usually not need this.",
        type=str,
    )
    a.add_argument(
        "--cost-model-network-config",
        default=None,
        help="Provide a network config as propagated by the cardano-node for the eval command.",
        type=str,
    )
    a.add_argument(
        "--eval-cpu-budget",
        default=None,
        help="Provide a CPU budget for the eval command.",
        type=int,
    )
    a.add_argument(
        "--eval-memory-budget",
        default=None,
        help="Provide a Memory budget for the eval command.",
        type=int,
    )
    a.add_argument(
        "--plutus-version",
        default=2,
        help="Plutus version to use.",
        choices=[1, 2],
        type=int,
    )
    for k, v in ARGPARSE_ARGS.items():
        alts = v.pop("__alts__", [])
        type = v.pop("type", None)
        if type is None:
            a.add_argument(
                f"-f{k.replace('_', '-')}",
                *alts,
                **v,
                action="store_true",
                dest=k,
                default=None,
            )
            a.add_argument(
                f"-fno-{k.replace('_', '-')}",
                action="store_false",
                help=argparse.SUPPRESS,
                dest=k,
                default=None,
            )
        else:
            a.add_argument(
                f"-f{k.replace('_', '-')}",
                *alts,
                **v,
                type=type,
                dest=k,
                default=None,
            )
    a.add_argument(
        "-O",
        default=1,
        type=int,
        help="The optimization level to use. Choose between 0 (nothing) and 3 (aggressive, semantics changing). Defaults to 1.",
        choices=range(len(OPT_CONFIGS)),
        dest="opt_level",
    )
    return a.parse_args()


def main():
    args = get_args()
    sys.setrecursionlimit(args.recursion_limit)

    # generate the compiler config
    compiler_config = DEFAULT_CONFIG
    compiler_config = compiler_config.update(OPT_CONFIGS[args.opt_level])
    overrides = {}
    for k in ARGPARSE_ARGS.keys():
        if getattr(args, k) is not None:
            overrides[k] = getattr(args, k)
    compiler_config = compiler_config.update(CompilationConfig(**overrides))

    command = Command(args.command)
    input_file = pathlib.Path(args.input_file) if args.input_file != "-" else sys.stdin
    with open(input_file, "r") as f:
        source_code = f.read()

    if args.from_cbor:
        code = unflatten(bytes.fromhex(source_code))
    elif args.from_hex:
        code = unflatten(cbor2.dumps(bytes.fromhex(source_code)))
    else:
        code: Program = parse(
            source_code,
            input_file.absolute() if isinstance(input_file, pathlib.Path) else None,
        )

    if command == Command.parse:
        print("Parsed successfully.")
        return

    code = compile(code, compiler_config)

    # Apply CLI parameters to code (i.e. to parameterize a parameterized contract)
    # UPLC lambdas may only take one argument at a time, so we evaluate by repeatedly applying
    code = apply(code, *map(lambda a: parse(f"(program 1.0.0 {a})").term, args.args))

    if command == Command.dump:
        print(dumps(code, UPLCDialect(args.dialect)))
        return

    if command == Command.build:
        if args.output_directory == "":
            if args.input_file == "-":
                print(
                    "Please supply an output directory if no input file is specified."
                )
                exit(-1)
            target_dir = pathlib.Path("build") / input_file.stem
        else:
            target_dir = pathlib.Path(args.output_directory)
        target_dir.mkdir(exist_ok=True, parents=True)
        cbor_bytes = flatten(code)
        cbor_hex = cbor_bytes.hex()
        # create cbor file for use with pycardano/lucid
        with (target_dir / "script.cbor").open("w") as fp:
            fp.write(cbor_hex)
        cbor = bytes.fromhex(cbor_hex)
        # double wrap
        cbor_wrapped = cbor2.dumps(cbor)
        cbor_wrapped_hex = cbor_wrapped.hex()
        # create plutus file
        d = {
            "type": "PlutusScriptV2",
            "description": f"",
            "cborHex": cbor_wrapped_hex,
        }
        with (target_dir / "script.plutus").open("w") as fp:
            json.dump(d, fp)
        script_hash = pycardano.plutus_script_hash(pycardano.PlutusV2Script(cbor))
        # generate policy ids
        with (target_dir / "script.policy_id").open("w") as fp:
            fp.write(script_hash.to_primitive().hex())
        addr_mainnet = pycardano.Address(
            script_hash, network=pycardano.Network.MAINNET
        ).encode()
        # generate addresses
        with (target_dir / "mainnet.addr").open("w") as fp:
            fp.write(addr_mainnet)
        addr_testnet = pycardano.Address(
            script_hash, network=pycardano.Network.TESTNET
        ).encode()
        with (target_dir / "testnet.addr").open("w") as fp:
            fp.write(addr_testnet)

        print(f"Wrote script artifacts to {target_dir}/")
        return
    if command == Command.eval:
        print("Starting execution")
        if args.builtin_cost_model_file is not None:
            with open(args.builtin_cost_model_file, "r") as fp:
                builtin_cost_model = parse_builtin_cost_model(json.load(fp))
        else:
            builtin_cost_model = default_builtin_cost_model_plutus_v2()
        if args.cek_machine_cost_model_file is not None:
            with open(args.cek_machine_cost_model_file, "r") as fp:
                cek_machine_cost_model = parse_cek_machine_cost_model(json.load(fp))
        else:
            cek_machine_cost_model = default_cek_machine_cost_model_plutus_v2()
        if args.cost_model_network_config is not None:
            with open(args.cost_model_network_config, "r") as fp:
                network_config = load_network_config(json.load(fp))
        else:
            network_config = latest_network_config()
        network_config = network_config[f"PlutusV{args.plutus_version}"]
        cek_machine_cost_model = updated_cek_machine_cost_model_from_network_config(
            cek_machine_cost_model, network_config
        )
        builtin_cost_model = updated_builtin_cost_model_from_network_config(
            builtin_cost_model, network_config
        )
        budget = default_budget()
        if args.eval_cpu_budget:
            budget.cpu = args.eval_cpu_budget
        if args.eval_memory_budget:
            budget.memory = args.eval_memory_budget
        ret = eval(
            code,
            budget=budget,
            cek_machine_cost_model=cek_machine_cost_model,
            builtin_cost_model=builtin_cost_model,
        )
        print("-------LOGS-------")
        if ret.logs:
            for line in ret.logs:
                print(line)
        else:
            print("None.")
        print("-------COST-------")
        print(f"CPU: {ret.cost.cpu}")
        print(f"Memory: {ret.cost.memory}")
        if isinstance(ret.result, Exception):
            print("-----ERROR-------")
            print(ret.result)
        else:
            print("-----SUCCESS-----")
            print(ret.result.dumps())


if __name__ == "__main__":
    main()
