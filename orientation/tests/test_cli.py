from orientation.cli import OrientationCLI

def run_cli(argv):
    cli = OrientationCLI()
    cli.run(argv)

def test_cli_matrix_from_axis_and_to_quat(capsys):
    run_cli(["matrix-from-axis", "--axis", "0", "0", "1", "--phi", "0.5"])
    out = capsys.readouterr().out.strip()
    assert out  # printed a matrix

    run_cli(["to-quat", "--matrix", "1","0","0","0","1","0","0","0","1"])
    out = capsys.readouterr().out.strip()
    # should print 4 float numbers (e0 e1 e2 e3)
    assert len(out.split()) == 4

def test_cli_matrix_to_rodrigues_identity(capsys):
    # Rodrigues of identity should be (0,0,0)
    from orientation.cli import OrientationCLI
    OrientationCLI().run(["matrix-to-rodrigues", "--matrix",
                          "1","0","0","0","1","0","0","0","1"])
    out = capsys.readouterr().out.strip()
    w = [float(x) for x in out.split()]
    assert len(w) == 3
    assert abs(w[0]) < 1e-12 and abs(w[1]) < 1e-12 and abs(w[2]) < 1e-12
