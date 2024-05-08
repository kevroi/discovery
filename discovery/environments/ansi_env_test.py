import gymnasium as gym

from discovery.environments import climbing


actions = {
    "W": 0,  # up
    "S": 1,  # anchor
}


def main():
    env = gym.make("discovery/Climbing-v0")
    print("Use the characters 'W', 'S'' or 'w', 's' to move the agent up or anchor in.")
    print()
    observation, info = env.reset(seed=42)
    print(env.render())
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
        print(env.render())
        print("obs: ", observation)
        print("info: ", info)
        print("reward, term, trunc, info:", reward, terminated, truncated)

        if terminated or truncated:
            observation, info = env.reset()
            print(env.render())
            print("obs: ", observation)
            print("info: ", info)

    env.close()


if __name__ == "__main__":
    main()
