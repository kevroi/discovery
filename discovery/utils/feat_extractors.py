from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

from discovery.utils.activations import CReLU, FTA
from discovery.utils.vqvae import VQVAEModel


class ClimbingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, include_anchor_bit: bool = False):
        self.include_anchor_bit = include_anchor_bit
        total_features_dim = observation_space["agent_loc"].n
        if self.include_anchor_bit:
            total_features_dim += 2
        super(ClimbingFeatureExtractor, self).__init__(
            observation_space, features_dim=total_features_dim
        )

    def forward(self, observations):
        if self.include_anchor_bit:
            features = torch.cat(
                (observations["agent_loc"], observations["at_anchor"]), dim=-1
            )
        else:
            features = torch.tensor(observations["agent_loc"], dtype=torch.float32)
        return features


# CNN from MiniGrid Documentation
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        last_layer_activation="relu",
    ) -> None:
        super().__init__(observation_space, features_dim)
        if isinstance(observation_space, spaces.Dict):
            observation_space = observation_space["image"]
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        if last_layer_activation == "relu":
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        elif last_layer_activation == "crelu":
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim // 2), CReLU()
            )
        elif last_layer_activation == "fta":
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim // 20), FTA())
        elif last_layer_activation == "lrelu":
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim), nn.LeakyReLU()
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class MinigridAutoEncoder(MinigridFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        last_layer_activation="relu",
    ) -> None:
        super().__init__(
            observation_space, features_dim, normalized_image, last_layer_activation
        )
        n_input_channels = observation_space.shape[0]

        # Compute shape by doing one forward pass
        with torch.no_grad():
            cnn_out_shape = self.cnn[:5](
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[2:]

        self.decoder = nn.Sequential(
            nn.Linear(features_dim, self.linear[0].in_features),
            nn.Unflatten(1, (64, cnn_out_shape[0], cnn_out_shape[1])),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, n_input_channels, (2, 2)),
        )

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def reconstruct(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.decode(self.forward(observations)), {}


class MinigridVQVAE(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        last_layer_activation="relu",
    ) -> None:
        super().__init__(observation_space, features_dim)
        if isinstance(observation_space, spaces.Dict):
            observation_space = observation_space["image"]
        n_input_channels = observation_space.shape[0]
        embedding_dim = 64

        encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, embedding_dim, (2, 2)),
            nn.ReLU(),
        )
        decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 32, (2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, n_input_channels, (2, 2)),
        )

        self.vqvae = VQVAEModel(
            obs_dim=observation_space.shape,
            codebook_size=128,
            embedding_dim=embedding_dim,
            encoder=encoder,
            decoder=decoder,
        )

        # Convert VQ-VAE discrete (torch.long) outputs to continuous-valued embedding vectors
        self.embeds = nn.Embedding(128, embedding_dim)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.embeds(
                self.vqvae.encode(
                    torch.as_tensor(observation_space.sample()[None]).float()
                )
            ).numel()

        if last_layer_activation == "relu":
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        elif last_layer_activation == "crelu":
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim // 2), CReLU()
            )
        elif last_layer_activation == "fta":
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim // 20), FTA())
        elif last_layer_activation == "lrelu":
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim), nn.LeakyReLU()
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        discrete_embeds = self.vqvae.encode(observations)
        embeds = self.embeds(discrete_embeds)
        return self.linear(torch.flatten(embeds, start_dim=1))

    def reconstruct(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.vqvae(observations)


class FeaturesWithHallwayExtractor(MinigridFeaturesExtractor):
    """Same as MinigridFeaturesExtractor but with an additional hallway feature.
    The hallway feature is a 2D one hot encoding of whether the agent is in the hallway or not.
    Note that this would require the agent to use a MultiInputPolicy, rather than a CnnPolicy.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        last_layer_activation="relu",
    ) -> None:
        super().__init__(
            observation_space,
            features_dim - 2,
            normalized_image,
            last_layer_activation,
        )

    def forward(self, observations: dict) -> torch.Tensor:
        features = super().forward(observations["image"])
        if len(observations["at_hallway"].shape) == 3:
            hallway = observations["at_hallway"].squeeze(1)
        else:
            assert len(observations["at_hallway"].shape) == 2
            hallway = observations["at_hallway"]
        outcome = torch.cat((features, hallway), dim=-1)
        return outcome

    @property
    def features_dim(self) -> int:
        return self._features_dim + 2  # Add 2 for the hallway feature.


class SharedPrivateFeaturesExtractor(MinigridFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        env: gym.Env,
        features_dim: int = 512,
        normalized_image: bool = False,
        last_layer_activation="relu",
    ) -> None:
        self.env = env
        self.private_feat_dim = 4
        self.num_shared = features_dim - self.private_feat_dim
        self.num_variants = env.num_variants
        self.private_feat_vecs = [
            torch.rand(self.private_feat_dim) for i in range(self.num_variants)
        ]
        for priv_vec in self.private_feat_vecs:
            priv_vec.requires_grad = True
        super().__init__(
            observation_space, features_dim, normalized_image, last_layer_activation
        )

    # def forward(self, observations: torch.Tensor) -> torch.Tensor:
    #     features = super().forward(observations)
    #     variant_idx = self.env.variant_idx  # Assuming env has a variant_idx attribute

    # split the features into shared and private
    # shared_features = features[:, :self.num_shared]
    # private_features = features[:, self.num_shared:]

    # features.register_hook(lambda x: self.modify_grad(x, [variant_idx]))

    # # detach the private features
    # for i in range(self.num_variants):
    #     if i == variant_idx:
    #         private_to_use = private_features[:, i]
    #     else:
    #         # private_features[:, i] = 0.0  # Zero out the private features
    #         masked_privates = torch.zeros_like(private_features[:, i])

    # # unsplit the features
    # features = torch.cat([shared_features, private_to_use.unsqueeze(1).detach(), ], dim=1)

    # for i in range(self.num_variants):
    #     if i == variant_idx and not features[:, self.num_shared+i].requires_grad:
    #         features[:, self.num_shared+i].detach.requires_grad_()  # Reattach the ith variant
    #     else:
    #         features[:, self.num_shared+i] = features[:, self.num_shared+i].detach()

    # # set private features to zero
    # self.set_mask(variant_idx)
    # features = features * self.mask.to(features.device)
    # top = features[:, :self.num_shared+variant_idx-1]
    # middle = features[:, self.num_shared+variant_idx-1].detach().unsqueeze(1)
    # if variant_idx == self.num_variants:
    #     features = torch.cat([top, middle], dim=1)
    # else:
    #     bottom = features[:, self.num_shared+variant_idx:]
    #     features = torch.cat([top, middle, bottom], dim=1)

    # return features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = super().forward(observations)
        features = features[:, : -self.private_feat_dim]

        variant_idx = self.env.variant_idx
        sp_feats = torch.cat(
            (
                features,
                self.private_feat_vecs[variant_idx - 1].repeat(features.shape[0], 1),
            ),
            dim=1,
        )
        print("variant idx:", variant_idx)
        # if variant_idx == 3:
        print("priv feat vec 1:", self.private_feat_vecs[variant_idx - 1])
        print("priv feat grad", self.private_feat_vecs[variant_idx - 1].grad)
        return sp_feats

    def modify_grad(self, x, inds):
        x[inds] = 0
        return x

    def set_mask(self, variant_idx):
        mask = torch.zeros(self.num_shared + self.num_variants)
        mask[: self.num_shared] = 1
        mask[self.num_shared + variant_idx - 1] = (
            1  # -1 because variant_idx is 1-indexed
        )
        self.mask = mask


class MaskedCNNFeaturesExtractor(MinigridFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        env: gym.Env,
        features_dim: int = 512,
        normalized_image: bool = False,
        last_layer_activation="relu",
    ) -> None:
        self.env = env
        self.num_shared = features_dim
        self.num_variants = env.num_variants
        self.mask = None
        super().__init__(
            observation_space,
            features_dim + self.num_variants,
            normalized_image,
            last_layer_activation,
        )

    def set_mask(self, variant_idx):
        mask = torch.zeros(self.num_shared + self.num_variants)
        mask[: self.num_shared] = 1
        mask[self.num_shared + variant_idx - 1] = (
            1  # -1 because variant_idx is 1-indexed
        )
        self.mask = mask

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = super().forward(observations)
        variant_idx = self.env.variant_idx
        # print("Variant idx:", variant_idx)
        self.set_mask(variant_idx)
        features = features * self.mask
        # features.register_hook(lambda grad: grad * self.mask)
        return features

    # def zero_grad(self):
    #     if self.mask is not None:
    #         self.mask.grad.zero_()

    # def backward(self, retain_graph=False):
    #     if self.mask is not None:
    #         self.mask.retain_grad()
    #         self.mask.grad = torch.where(self.mask > 0.0, self.mask.grad, torch.zeros_like(self.mask.grad))
    #     super().backward(retain_graph=retain_graph)

    # def backward(self, retain_graph=False):
    #     if self.mask is not None:
    #         # Retain the gradient of the mask
    #         self.mask.retain_grad()
    #         # Copy the original mask gradients for comparison
    #         mask_grad_orig = self.mask.grad.clone()
    #         # Update the mask gradients based on the mask values
    #         self.mask.grad = torch.where(self.mask > 0.0, self.mask.grad, torch.zeros_like(self.mask.grad))
    #         # Calculate the change in gradients for masked features
    #         mask_grad_change = self.mask.grad - mask_grad_orig
    #         # Print out the change in gradients
    #         print("Change in gradients for masked features:", mask_grad_change)
    #     # Call the base class's backward() method
    #     breakpoint()
    #     super().backward(retain_graph=retain_graph)

    # def detach_mask(self):
    #     if self.mask is not None:
    #         self.mask = self.mask.detach().clone().requires_grad_(True)


# CNN from DQN Nature paper
class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        last_layer_activation="relu",
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(
            observation_space, check_channels=False, normalized_image=normalized_image
        ), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        if last_layer_activation == "relu":
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        elif last_layer_activation == "crelu":
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim // 2), CReLU()
            )
        elif last_layer_activation == "fta":
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim // 20), FTA())
        elif last_layer_activation == "lrelu":
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim), nn.LeakyReLU()
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
