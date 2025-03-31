from urdf_parser_py.urdf import URDF
robot = URDF.from_xml_file("/home/erc/RL_NVIDIA/IsaacGymEnvs/assets/urdf/gogoro/urdf/scooter_V14.urdf")
print("Root link:", robot.get_root())
