import os
import subprocess
from pathlib import Path

from tests.test_acceptance import acceptance_test_dirs

for dirpath in acceptance_test_dirs():
    files = os.listdir(dirpath)
    input_file = Path(dirpath).joinpath(next(f for f in files if f.endswith("uplc")))
    output_file = Path(dirpath).joinpath(f"{Path(input_file).stem}.cost")
    res = subprocess.run(["aiken", "uplc", "eval", input_file], stdout=subprocess.PIPE)
    if res.returncode != 0:
        output = "error"
    else:
        output = res.stdout.decode("utf8")
    with open(output_file, "w") as f:
        f.write(output)
