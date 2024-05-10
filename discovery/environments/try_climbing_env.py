import argparse
import re
import gymnasium as gym
from matplotlib import pyplot as plt

from discovery.environments import climbing


parser = argparse.ArgumentParser()
parser.add_argument(
    "--render_mode",
    choices=["ansi", "rgb_array"],
    type=str,
    default="ansi",
    required=False,
    help="How to render.",
)

actions = {
    "W": 0,  # up
    "S": 1,  # anchor
}


class AnsiRenderer:

    def __call__(self, rendered_output):
        print(rendered_output)


class RgBArrayRenderer:

    def __init__(self) -> None:
        plt.ion()
        # self._first = True

    def __call__(self, rendered_output):
        plt.imshow(rendered_output)
        plt.draw()
        plt.show()
        # if self._first:
        #     self._first = False
        #     plt.show()
        plt.pause(0.1)


def main():
    args = parser.parse_args()
    env = gym.make("discovery/Climbing-v0", render_mode=args.render_mode)

    if args.render_mode == "ansi":
        env_renderer = AnsiRenderer()
    elif args.render_mode == "rgb_array":
        env_renderer = RgBArrayRenderer()
    else:
        assert False, f"Unknown render mode: {args.render_mode}"

    print("Use the characters 'W', 'S'' or 'w', 's' to move the agent up or anchor in.")
    print()
    observation, info = env.reset(seed=42)
    env_renderer(env.render())
    print("obs: ", observation)
    print("info: ", info)
    for _ in range(1000):
        # action = env.action_space.sample()
        action = None
        while action is None:
            action = input("action: ").strip().upper()
            if action not in actions:
                print(f"Invalid action: {action}")
                action = None
            else:
                action = actions[action]
        observation, reward, terminated, truncated, info = env.step(action)
        env_renderer(env.render())
        print("obs: ", observation)
        print("info: ", info)
        print("reward, term, trunc, info:", reward, terminated, truncated)

        if terminated or truncated:
            observation, info = env.reset()
            env_renderer(env.render())
            print("obs: ", observation)
            print("info: ", info)

    env.close()


if __name__ == "__main__":
    main()
