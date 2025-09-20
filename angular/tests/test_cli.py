import numpy as np
from angular.cli import main

def test_cli_from_euler_smoke(capsys):
    main(["from-euler", "--order", "Z", "--angles", "0,0.0,0"])
    out = capsys.readouterr().out
    assert "[[1." in out and "0." in out
