import numpy as np
from orientation.apis import OrientationService
from orientation.core import SO3

def test_euler_in_out_orders():
    svc = OrientationService()
    ang = [0.2, -0.3, 0.4]
    for order in ["ZYX", "XYZ"]:
        R = svc.euler_to_matrix(ang, order=order, degrees=False)
        est = svc.matrix_to_euler(R, order=order, degrees=False)
        # Allow a little slop since GN may stop with tiny residual
        assert np.allclose(est, np.array(ang), atol=1e-5)

def test_expmap_equivalence():
    svc = OrientationService()
    u = np.array([0.2, -0.1, 0.5]); u = u/np.linalg.norm(u)
    phi = 0.9
    R1 = svc.expmap(u*phi)
    R2 = SO3.from_axis_angle(phi, u).R
    assert np.allclose(R1, R2, atol=1e-9)

def test_random_so3_shapes_and_orthogonality():
    svc = OrientationService()
    mats = svc.random_so3(5)
    assert len(mats) == 5
    for R in mats:
        RtR = R.T @ R
        assert np.allclose(RtR, np.eye(3), atol=1e-10)
