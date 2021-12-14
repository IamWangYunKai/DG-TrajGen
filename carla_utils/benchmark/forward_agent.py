from carla_utils import carla, agents

from ..system import env_path
env_path.append(env_path.carla_api_path)
Agent = agents.navigation.agent.Agent


class ForwardAgent(Agent):
    """
    Simple derivation of Agent Class,
    A trivial agent agent that goes straight
    """
    def run_step(self, measurements, sensor_data, directions, target):
        control = carla.VehicleControl()
        control.throttle = 0.9

        return control
