from applied_dynamics.cli import main

def test_smoke_cli_commands(monkeypatch, tmp_path):
    for cmd in ("pendulum", "spherical", "planar2r", "absorber"):
        assert main([cmd]) == 0
