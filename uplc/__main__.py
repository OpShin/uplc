import argparse
import enum
import json
import pathlib
import sys

import cbor2
import pyaiken
import pycardano

from .util import *
from .ast import Program, Apply
from .transformer import unique_variables


class Command(enum.Enum):
    eval = "eval"
    parse = "parse"
    build = "build"
    dump = "dump"


def main():
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
        default=UPLCDialect.Aiken.value,
        help="The dialect for dumping the parsed UPLC.",
        choices=[d.value for d in UPLCDialect],
    )
    a.add_argument(
        "--unique-varnames",
        action="store_true",
        help="Assign variables a unique name.",
    )
    a.add_argument(
        "--from-cbor",
        action="store_true",
        help="Read hex representation of flattened UPLC.",
    )
    a.add_argument(
        "args",
        nargs="*",
        default=[],
        help="Input parameters for the function, in case the command is eval.",
    )
    args = a.parse_args()
    command = Command(args.command)
    input_file = pathlib.Path(args.input_file) if args.input_file != "-" else sys.stdin
    with open(input_file, "r") as f:
        source_code = f.read()

    if args.from_cbor:
        source_code = pyaiken.uplc.unflat(source_code)
    code = parse(
        source_code,
        input_file.absolute() if isinstance(input_file, pathlib.Path) else None,
    )

    if command == Command.parse:
        print("Parsed successfully.")
        return

    if args.unique_varnames:
        code = unique_variables.UniqueVariableTransformer().visit(code)

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
            target_dir = pathlib.Path(pathlib.Path(input_file).stem)
        else:
            target_dir = pathlib.Path(args.output_directory)
        target_dir.mkdir(exist_ok=True)
        uplc_dump = code.dumps(dialect=UPLCDialect.Aiken)
        cbor_hex = pyaiken.uplc.flat(uplc_dump)
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
        print("------------------")
        assert isinstance(
            code, Program
        ), "Main function must be wrapped in (program 1.0.0 ...)"
        try:
            f = code.term
            # UPLC lambdas may only take one argument at a time, so we evaluate by repeatedly applying
            for d in map(lambda a: parse(f"(program 1.0.0 {a})").term, args.args):
                f = Apply(f, d)
            ret = eval(f).dumps()
        except Exception as e:
            print("An exception was raised")
            ret = e
        print("------------------")
        print(ret)


if __name__ == "__main__":
    main()
