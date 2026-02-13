import unittest

from robotic_grounding.tests.utils import APP_IS_READY

if APP_IS_READY:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from robotic_grounding.assets import OBJECTS_ASSET_DIR

    # Import the robot configurations
    from robotic_grounding.assets.g1 import G1_CYLINDER_CFG


@configclass
class LoadAssetSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with G1 robot."""

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="/World/Robot")

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{OBJECTS_ASSET_DIR}/apple/apple.usd",
        ),
    )


@unittest.skipIf(not APP_IS_READY, "App is not ready")
class TestLoadAsset(unittest.TestCase):
    """Test that robot assets can be instantiated correctly."""

    def setUp(self) -> None:
        """Set up a simulation context for each test."""
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))

        # Create scene with G1 robot and object
        scene_cfg = LoadAssetSceneCfg(num_envs=1, env_spacing=2.0)
        self.scene = InteractiveScene(scene_cfg)

    def tearDown(self) -> None:
        """Clean up after each test."""
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    def test_instantiate_g1_robot(self) -> None:
        """Test that G1 robot can be instantiated as an Articulation."""
        robot = self.scene["robot"]
        self.assertIsNotNone(robot, "Robot not found")

    def test_instantiate_object(self) -> None:
        """Test that object can be instantiated as a RigidObject."""
        object = self.scene["object"]
        self.assertIsNotNone(object, "Object not found")


if __name__ == "__main__":
    unittest.main()
