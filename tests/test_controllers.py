import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from dpl_mppi import DPLMPPIController
from gl_mppi import GLMPPIController
from nl_mppi import NLMPPIController
from pl_mppi import PLMPPIController
from sm_mppi import SMMPPIController
from utils import dynamics


CONTROLLERS = {
    "sm": SMMPPIController,
    "nl": NLMPPIController,
    "gl": GLMPPIController,
    "pl": PLMPPIController,
    "dpl": DPLMPPIController,
}
FIXTURES = json.loads(Path(__file__).with_name("controller_costs.json").read_text())


class ControllerTests(unittest.TestCase):
    def test_cost_regression(self) -> None:
        fixtures = FIXTURES
        for name, case in fixtures["cases"].items():
            with self.subTest(controller=name):
                controller_class = CONTROLLERS[case["algorithm"]]
                options = (
                    {"passing_side": case["side"]}
                    if case["algorithm"] == "pl"
                    else {}
                )
                if case["algorithm"] == "gl":
                    options = {"hallway_goals": case["hallway_goals"]}
                controller = controller_class(
                    fixtures["static_obstacles"],
                    "cpu",
                    goal=(0.0, -1.5),
                    config=replace(
                        controller_class.default_config, num_samples=3, horizon=4
                    ),
                    **options,
                )
                controller.robot_velocity = torch.tensor([0.03, -0.3])
                controller.agent_velocities = {0: torch.tensor([0.05, 0.3])}
                for step in case["steps"]:
                    agent = torch.tensor(step["agent"])
                    controller.agent_predictions = {
                        0: agent[:2]
                        + controller.prediction_times[:, None]
                        * controller.agent_velocities[0]
                    }
                    controller.interacting_agents = {0} if agent[1] < 1.4 else set()
                    state = torch.tensor(fixtures["states"]) + torch.tensor(
                        step["shift"]
                    )
                    cost = controller.terminal_cost(state, torch.zeros(1, 3, 4, 2))
                    torch.testing.assert_close(
                        cost,
                        torch.tensor(step["costs"]),
                        rtol=1e-6,
                        atol=1e-4,
                    )
                    torch.testing.assert_close(
                        controller.goal, torch.tensor(step["goal"])
                    )

    def test_mppi_control(self) -> None:
        for name, controller_class in CONTROLLERS.items():
            with self.subTest(controller=name):
                torch.manual_seed(17)
                options = {"passing_side": -1} if name == "pl" else {}
                if name == "gl":
                    options = {"hallway_goals": ((0.0, -1.5), (0.0, 1.5))}
                controller = controller_class(
                    FIXTURES["static_obstacles"], "cpu", goal=(0.0, -1.5), **options
                )
                for step in range(3):
                    action, rollouts, costs, reached = controller.compute_control(
                        torch.tensor([0.0, 1.4 - 0.05 * step, -torch.pi / 2]),
                        torch.tensor([0.0, -0.3]),
                        {7: torch.tensor([0.2, 0.4, torch.pi / 2])},
                        {7: torch.tensor([0.0, 0.3])},
                    )
                    self.assertEqual(action.shape, (2,))
                    self.assertEqual(rollouts.shape, (250, 10, 3))
                    self.assertEqual(costs.shape, (250,))
                    self.assertFalse(reached)
                    self.assertTrue(torch.isfinite(costs).all())
                    self.assertTrue(torch.isfinite(rollouts).all())
                    self.assertTrue((action >= controller.mppi.u_min - 1e-6).all())
                    self.assertTrue((action <= controller.mppi.u_max + 1e-6).all())

    def test_goal_update_preserves_optimizer_and_history(self) -> None:
        controller = PLMPPIController([], "cpu", goal=(0.0, -1.5), passing_side=1)
        controller.history.fill_(0.5)
        controls = controller.mppi.U.clone()
        controller.set_goal((0.0, 1.5))
        torch.testing.assert_close(controller.goal, torch.tensor([0.15, 1.5]))
        torch.testing.assert_close(
            controller.other_goals, torch.tensor([[-0.15, 1.5], [0.0, 1.5]])
        )
        torch.testing.assert_close(controller.mppi.U, controls)
        self.assertTrue((controller.history == 0.5).all())

    def test_gl_hallway_endpoints(self) -> None:
        endpoints = ((-2.0, 1.0), (3.0, 4.0))
        controller = GLMPPIController(
            [], "cpu", goal=endpoints[0], hallway_goals=endpoints
        )
        torch.testing.assert_close(
            controller.candidate_goals, torch.tensor(endpoints)
        )
        torch.testing.assert_close(
            controller.other_goals, torch.tensor([endpoints[1]])
        )
        controller.history.fill_(0.5)
        controls = controller.mppi.U.clone()
        controller.set_goal(endpoints[1])
        torch.testing.assert_close(controller.goal, torch.tensor(endpoints[1]))
        torch.testing.assert_close(
            controller.other_goals, torch.tensor([endpoints[0]])
        )
        torch.testing.assert_close(controller.mppi.U, controls)
        self.assertTrue((controller.history == 0.5).all())

    def test_intent_objective_multiplier(self) -> None:
        for controller_class in (
            GLMPPIController, PLMPPIController, DPLMPPIController
        ):
            with self.subTest(controller=controller_class.__name__):
                options = {}
                if controller_class is GLMPPIController:
                    options = {"hallway_goals": ((0.0, 0.0), (0.0, 3.0))}
                elif controller_class is PLMPPIController:
                    options = {"passing_side": -1}
                controller = controller_class(
                    [],
                    "cpu",
                    goal=(0.0, 0.0),
                    config=replace(
                        controller_class.default_config,
                        num_samples=1,
                        horizon=1,
                        history_length=1,
                    ),
                    **options,
                )
                if controller_class is GLMPPIController:
                    position = torch.tensor([0.0, 0.0])
                    probability = 1 / (1 + math.exp(-3))
                else:
                    offset = 0.15 if controller_class is PLMPPIController else 0.2
                    position = torch.tensor([-offset, 0.0])
                    probability = 1 / (
                        1 + math.exp(-2 * offset) + math.exp(-offset)
                    )
                    controller.interacting_agents = {0}
                    controller.agent_predictions = {0: torch.tensor([[2.0, 0.0]])}
                state = torch.tensor([[[[float(position[0]), 0.0, 0.0]]]])
                cost = controller.terminal_cost(state, torch.zeros(1, 1, 1, 2))
                torch.testing.assert_close(
                    cost, torch.tensor([-500 * probability]), rtol=1e-6, atol=1e-4
                )

    def test_goal_reached(self) -> None:
        controller = NLMPPIController([], "cpu", goal=(0.0, 0.0))
        _, _, _, reached = controller.compute_control(
            torch.zeros(3),
            torch.zeros(2),
            {0: torch.tensor([0.0, 2.0, 0.0])},
            {0: torch.zeros(2)},
        )
        self.assertTrue(reached)

    def test_unicycle_dynamics(self) -> None:
        state = torch.tensor([[1.0, 2.0, 0.0], [1.0, 2.0, math.pi / 2]])
        action = torch.tensor([[0.4, 0.0], [0.4, 1.0]])
        expected = torch.tensor(
            [
                [1.12, 2.0, 0.0],
                [
                    1.0 - 0.4 * (1 - math.cos(0.3)),
                    2.0 + 0.4 * math.sin(0.3),
                    math.pi / 2 + 0.3,
                ],
            ]
        )
        torch.testing.assert_close(dynamics(state, action, 0.3), expected)


if __name__ == "__main__":
    unittest.main()
