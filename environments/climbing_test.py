import unittest
import gymnasium as gym
import numpy as np

from environments import climbing


class TestClimbingEnv(unittest.TestCase):

    def test_reset(self):
        env = gym.make("discovery/Climbing-v0", height=4, anchor_interval=2)
        obs, info = env.reset()
        # Check the start conditions.
        self.assertEqual(obs["agent_loc"], 0)
        self.assertEqual(obs["at_anchor"], 0)
        self.assertEqual(obs["last_anchor_loc"], 0)

        env.step(climbing.DEFAULT_UP_ACTION)
        obs, _, _, _, _, = env.step(climbing.DEFAULT_UP_ACTION)  # fmt: skip
        # Should be at anchor -- this is not what we are testing though.
        assert obs["at_anchor"] == 1
        # Anchor in.
        env.step(climbing.DEFAULT_ANCHOR_ACTION)
        obs, _, _, _, _, = env.step(climbing.DEFAULT_UP_ACTION)  # fmt: skip
        # Now all state should be different from the start, check this.
        self.assertNotEqual(obs["agent_loc"], 0)
        self.assertNotEqual(obs["last_anchor_loc"], 0)

        # Now test reset.
        obs, info = env.reset()
        self.assertEqual(obs["agent_loc"], 0)
        self.assertEqual(obs["at_anchor"], 0)
        self.assertEqual(obs["last_anchor_loc"], 0)

    def test_anchoring(self):
        env = gym.make("discovery/Climbing-v0", height=4, anchor_interval=2)
        first_anchor_loc = 2

        obs, info = env.reset()
        obs, _, _, _, _ = env.step(climbing.DEFAULT_UP_ACTION)
        obs, _, _, _, _ = env.step(climbing.DEFAULT_UP_ACTION)
        # Should be at anchor.
        assert obs["at_anchor"] == 1
        # Anchor in.
        env.step(climbing.DEFAULT_ANCHOR_ACTION)
        # Climb higher.
        env.step(climbing.DEFAULT_UP_ACTION)  # fmt: skip
        # Fall.
        obs, _, _, _, _, = env.step(climbing.DEFAULT_ANCHOR_ACTION)  # fmt: skip
        # Fall is just to the anchor.
        self.assertEqual(obs["agent_loc"], first_anchor_loc)
        self.assertEqual(obs["at_anchor"], 1)
        self.assertEqual(obs["last_anchor_loc"], first_anchor_loc)

    def test_not_anchoring(self):
        env = gym.make("discovery/Climbing-v0", height=4, anchor_interval=2)
        first_anchor_loc = 2

        obs, info = env.reset()
        obs, _, _, _, _ = env.step(climbing.DEFAULT_UP_ACTION)
        obs, _, _, _, _ = env.step(climbing.DEFAULT_UP_ACTION)
        # Should be at anchor -- but we keep climbing.
        assert obs["at_anchor"] == 1
        # Climb higher.
        env.step(climbing.DEFAULT_UP_ACTION)  # fmt: skip
        # Fall.
        obs, _, _, _, _, = env.step(climbing.DEFAULT_ANCHOR_ACTION)  # fmt: skip
        # Fall is all the way down.
        self.assertEqual(obs["agent_loc"], 0)
        self.assertEqual(obs["at_anchor"], 0)
        self.assertEqual(obs["last_anchor_loc"], 0)

    def test_randomized_actions(self):
        # TODO
        pass

    def test_reward_sequence(self):
        env = gym.make("discovery/Climbing-v0", height=4, anchor_interval=2)
        obs, info = env.reset()
        rewards = []
        for _ in range(3):  # Started at 0, so 3 more steps to reach the top.
            obs, reward, _, _, _ = env.step(climbing.DEFAULT_UP_ACTION)
            rewards.append(reward)
        # The agent should have reached the top.
        self.assertEqual(rewards, [0, 0, 1])
