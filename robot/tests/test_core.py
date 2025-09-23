import numpy as np
from robot.core import Link, Joint, RobotModel

def test_robot_model_shapes():
    links = [
        Link("L1", 1.0, np.array([0.5,0,0]), np.eye(3)),
        Link("L2", 1.0, np.array([0.5,0,0]), np.eye(3)),
    ]
    joints = [Joint("J1","R"), Joint("J2","R")]
    rob = RobotModel("R", links, joints)
    assert rob.dof == 2
    assert rob.masses().tolist() == [1.0, 1.0]