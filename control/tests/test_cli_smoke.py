from click.testing import CliRunner
from control.cli import app

def test_cli_smoke_msdpd_and_diagram(tmp_path, monkeypatch):
    runner = CliRunner()
    res = runner.invoke(app, ["diagram", "--out", "classes_test"])
    assert res.exit_code == 0
    res = runner.invoke(app, ["msd_pd","--T","0.1"])
    assert res.exit_code == 0
