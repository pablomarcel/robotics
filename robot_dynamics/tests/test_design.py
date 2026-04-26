from robot_dynamics.design import DHChainBuilder

def test_dh_builder_planar2r(planar2r_model):
    assert planar2r_model.name == "Planar2R"
    assert planar2r_model.dof == 2