# =============================
# File: rotation_kinematics/rot_design.py
# =============================
"""Design-time constants and small helpers for the CLI.
Covers: the 12 global (Tait–Bryan) and 6 proper Euler sequences, local/global mapping,
and a few named presets from the chapter examples."""

VALID_TB = ["xyz","xzy","yxz","yzx","zxy","zyx"]
VALID_PROPER = ["zxz","xyx","yzy","zyz","xzx","yxy"]
VALID_ALL = VALID_TB + VALID_PROPER

# Named presets that mirror textbook examples
PRESETS = {
    # Example 19/20: roll–pitch–yaw in global axes
    "rpy_global": {"mode": "global", "seq": "xyz", "angles_deg": [0.0, 0.0, 0.0]},
}

HELP_SEQ = (
    "Sequence of axes. Tait–Bryan: xyz,xzy,yxz,yzx,zxy,zyx. Proper Euler: zxz,xyx,yzy,zyz,xzx,yxy."
)