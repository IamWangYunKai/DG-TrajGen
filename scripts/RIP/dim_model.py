from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)

class MLP(nn.Module):
    """A simple multi-layer perceptron module."""
    def __init__(
        self,
        input_size: int,
        output_sizes,
        activation_fn=nn.ReLU,
        dropout_rate=None,
        activate_final=False,
    ):
        super(MLP, self).__init__()

        layers = list()
        for in_features, out_features in zip(
            [input_size] + list(output_sizes)[:-2],
                output_sizes[:-1],
        ):
            # Fully connected layer.
            layers.append(nn.Linear(in_features, out_features))
            # Activation layer.
            layers.append(activation_fn(inplace=True))
            # (Optional) dropout layer.
            if dropout_rate is not None:
                layers.append(nn.Dropout(p=dropout_rate, inplace=True))
        # Final layer.
        layers.append(nn.Linear(output_sizes[-2], output_sizes[-1]))
        # (Optional) output activation layer.
        if activate_final:
            layers.append(activation_fn(inplace=True))

        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from the MLP."""
        return self._model(x)

class CNN(nn.Module):
    def __init__(self,input_dim=1, out_dim=256, bn=False):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        self.bn = bn
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn: x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        if self.bn: x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        if self.bn: x = self.bn3(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = x.view(-1, self.out_dim)
        return x

class MobileNetV2(nn.Module):
    def __init__(self,num_classes: int, in_channels: int = 3,):
        super(MobileNetV2, self).__init__()

        self._model = torch.hub.load(
            github="pytorch/vision:v0.6.0",
            model="mobilenet_v2",
            num_classes=num_classes,
        )

        # HACK(filangel): enables non-RGB visual features.
        _tmp = self._model.features._modules['0']._modules['0']
        self._model.features._modules['0']._modules['0'] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size,
            stride=_tmp.stride,
            padding=_tmp.padding,
            bias=_tmp.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from the MobileNetV2."""
        return self._model(x)


class AutoregressiveFlow(nn.Module):
    def __init__(self, output_shape, hidden_size: int = 64):
        """
        Args:
          output_shape: The shape of the base and data distribution (a.k.a. event_shape).
          hidden_size: The dimensionality of the GRU hidden state.
        """
        super(AutoregressiveFlow, self).__init__()
        self._output_shape = output_shape

        # Initialises the base distribution.
        self._base_dist = D.MultivariateNormal(
            loc=torch.zeros(self._output_shape[-2] * self._output_shape[-1]),
            scale_tril=torch.eye(self._output_shape[-2] *
                                 self._output_shape[-1]),
        )

        # The decoder recurrent network used for the sequence generation.
        self._decoder = nn.GRUCell(
            input_size=self._output_shape[-1],
            hidden_size=hidden_size,
        )

        # The output head.
        self._locscale = MLP(
            input_size=hidden_size,
            # output_sizes=[32, self._output_shape[0]],
            output_sizes=[32, 4],
            activation_fn=nn.ReLU,
            dropout_rate=None,
            activate_final=False,
        )

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self._base_dist = D.MultivariateNormal(
            loc=self._base_dist.mean.to(*args, **kwargs),
            scale_tril=self._base_dist.scale_tril.to(*args, **kwargs),
        )
        return self

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass, stochastic generation of a sequence.
        Args:
          z: The contextual parameters of the conditional density estimator, with
            shape `[B, K]`.

        Returns:
          The sampels from the push-forward distribution, with shape `[B, D]`.
        """
        # Samples from the base distribution.
        x = self._base_dist.sample_n(n=z.shape[0])
        x = x.reshape(-1, *self._output_shape)

        return self._forward(x, z)[0]

    def _forward(self, x: torch.Tensor, z: torch.Tensor):
        """Transforms samples from the base distribution to the data distribution.
        Args:
          x: Samples from the base distribution, with shape `[B, D]`.
          z: The contextual parameters of the conditional density estimator, with
            shape `[B, K]`.

        Returns:
          y: The sampels from the push-forward distribution,
            with shape `[B, D]`.
          logabsdet: The log absolute determinant of the Jacobian,
            with shape `[B]`.
        """
        # x: (B, D, 2)
        # Output containers.
        y = list()
        scales = list()

        # Initial input variable.
        y_tm1 = torch.zeros(
            size=(z.shape[0], self._output_shape[-1]),
            dtype=z.dtype,
        ).to(z.device)  # (B, 2)

        for t in range(x.shape[-2]):  # D
            x_t = x[:, t, :]  # (B, D, 2) - > (B, 2)
            # Unrolls the GRU.
            z = self._decoder(y_tm1, z)  # (B, 64)
            # Predicts the location and scale of the MVN distribution.
            dloc_scale = self._locscale(z)  # (B, 4)
            dloc = dloc_scale[..., :2]  # (B, 2)
            scale = F.softplus(dloc_scale[..., 2:]) + 1e-3  # (B, 2)

            # Data distribution corresponding sample.
            y_t = (y_tm1 + dloc) + scale * x_t

            # Update containers.
            y.append(y_t)
            scales.append(scale)
            y_tm1 = y_t

        # Prepare tensors, reshape to [B, D, 2].
        y = torch.stack(y, dim=-2)
        scales = torch.stack(scales, dim=-2)

        # Log absolute determinant of Jacobian.
        logabsdet = torch.log(torch.abs(torch.prod(scales, dim=-2)))
        logabsdet = torch.sum(logabsdet, dim=-1)

        return y, logabsdet

    def _inverse(self, y: torch.Tensor, z: torch.Tensor):  # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transforms samples from the data distribution to the base distribution.
        Args:
          y: Samples from the data distribution, with shape `[B, D]`.
          z: The contextual parameters of the conditional density estimator, with shape
            `[B, K]`.

        Returns:
          x: The sampels from the base distribution,
            with shape `[B, D]`.
          log_prob: The log-likelihood of the samples under
            the base distibution probability, with shape `[B]`.
          logabsdet: The log absolute determinant of the Jacobian,
            with shape `[B]`.
        """

        # Output containers.
        x = list()
        scales = list()

        # Initial input variable.
        y_tm1 = torch.zeros(
            size=(z.shape[0], self._output_shape[-1]),
            dtype=z.dtype,
        ).to(z.device)

        for t in range(y.shape[-2]):
            y_t = y[:, t, :]

            # Unrolls the GRU.
            z = self._decoder(y_tm1, z)

            # Predicts the location and scale of the MVN distribution.
            dloc_scale = self._locscale(z)
            dloc = dloc_scale[..., :2]
            scale = F.softplus(dloc_scale[..., 2:]) + 1e-3

            # Base distribution corresponding sample.
            x_t = (y_t - (y_tm1 + dloc)) / scale

            # Update containers.
            x.append(x_t)
            scales.append(scale)
            y_tm1 = y_t

        # Prepare tensors, reshape to [B, T, 2].
        x = torch.stack(x, dim=-2)
        scales = torch.stack(scales, dim=-2)

        # Log likelihood under base distribution.
        log_prob = self._base_dist.log_prob(x.view(x.shape[0], -1))

        # Log absolute determinant of Jacobian.
        logabsdet = torch.log(torch.abs(torch.prod(
            scales, dim=-1)))  # determinant == product over xy-coordinates
        logabsdet = torch.sum(logabsdet, dim=-1)  # sum over T dimension

        return x, log_prob, logabsdet


class ImitativeModel(nn.Module):
    def __init__(self, output_shape=(4, 2)):
        """
          output_shape: The shape of the base and data distribution (a.k.a. event_shape).
        """
        super(ImitativeModel, self).__init__()
        self._output_shape = output_shape

        # The convolutional encoder model.
        # self._encoder = MobileNetV2(num_classes=128, in_channels=6)
        self._encoder = CNN(input_dim=6, out_dim=128, bn=True)

        # Merges the encoded features and the vector inputs.
        self.mlp = MLP(
            input_size=128 + 1,
            output_sizes=[64, 64, 64],
            activation_fn=nn.ReLU,
            dropout_rate=None,
            activate_final=True,
        )

        # The decoder recurrent network used for the sequence generation.
        self._decoder = AutoregressiveFlow(
            output_shape=self._output_shape,
            hidden_size=64,
        )

    def to(self, *args, **kwargs):
        """Handles non-parameter tensors when moved to a new device."""
        self = super().to(*args, **kwargs)
        self._decoder = self._decoder.to(*args, **kwargs)
        return self

    def forward(
        self,
        num_steps: int,
        #goal: Optional[torch.Tensor] = None,
        lr: float = 1e-1,
        #epsilon: float = 1.0,
        **context: torch.Tensor
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """Returns a local mode from the posterior.
        Args:
          num_steps: The number of gradient-descent steps for finding the mode.
          goal: The locations of the the goals.
          epsilon: The tolerance parameter for the goal.
          context: (keyword arguments) The conditioning
            variables used for the conditional flow.

        Returns:
          A mode from the posterior, with shape `[D, 2]`.
        """
        if not "visual_features" in context:
            raise ValueError("Missing `visual_features` keyword argument.")
        batch_size = context["visual_features"].shape[0]

        # Sets initial sample to base distribution's mean.
        x = self._decoder._base_dist.sample().clone().detach().repeat(
            batch_size, 1).view(
                batch_size,
                *self._output_shape,
            )
        x.requires_grad = True

        # The contextual parameters, caches for efficiency.
        z = self._params(**context)

        # Initialises a gradient-based optimiser.
        optimizer = optim.Adam(params=[x], lr=lr)

        # Stores the best values.
        x_best = x.clone()
        loss_best = torch.ones(()).to(x.device) * 1000.0

        for _ in range(num_steps):
            # Resets optimizer's gradients.
            optimizer.zero_grad()
            # Operate on `y`-space.
            y, _ = self._decoder._forward(x=x, z=z)
            # Calculates imitation prior.
            _, log_prob, logabsdet = self._decoder._inverse(y=y, z=z)
            imitation_prior = torch.mean(log_prob - logabsdet)
            # Calculates goal likelihodd.
            goal_likelihood = 0.0
            # if goal is not None:
            #     goal_likelihood = self._goal_likelihood(y=y, goal=goal, epsilon=epsilon)
            loss = -(imitation_prior + goal_likelihood)
            # Backward pass.
            loss.backward(retain_graph=True)
            # Performs a gradient descent step.
            optimizer.step()
            # Book-keeping
            if loss < loss_best:
                x_best = x.clone()
                loss_best = loss.clone()

        y, _ = self._decoder._forward(x=x_best, z=z)

        return y

    def _goal_likelihood(self, y: torch.Tensor, goal: torch.Tensor, **hyperparams) -> torch.Tensor:
        """Returns the goal-likelihood of a plan `y`, given `goal`.
        Args:
          y: A plan under evaluation, with shape `[B, T, 2]`.
          goal: The goal locations, with shape `[B, K, 2]`.
          hyperparams: (keyword arguments) The goal-likelihood hyperparameters.

        Returns:
          The log-likelihodd of the plan `y` under the `goal` distribution.
        """
        # Parses tensor dimensions.
        B, K, _ = goal.shape

        # Fetches goal-likelihood hyperparameters.
        epsilon = hyperparams.get("epsilon", 1.0)

        # TODO(filangel): implement other goal likelihoods from the DIM paper
        # Initializes the goal distribution.
        goal_distribution = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(
                probs=torch.ones((B, K)).to(goal.device)),
            component_distribution=D.Independent(
                D.Normal(loc=goal, scale=torch.ones_like(goal) * epsilon),
                reinterpreted_batch_ndims=1,
            ))

        return torch.mean(goal_distribution.log_prob(y[:, -1, :]), dim=0)

    def _params(self, **context: torch.Tensor) -> torch.Tensor:
        if not "visual_features" in context:
            raise ValueError("Missing `visual_features` keyword argument.")
        if not "velocity" in context:
            raise ValueError("Missing `velocity` keyword argument.")

        visual_features = context.get("visual_features")
        velocity = context.get("velocity")

        # Encodes the visual input.

        visual_features = self._encoder(visual_features)

        # Merges visual input logits and vector inputs.
        visual_features = torch.cat(
            tensors=[
                visual_features,
                velocity,
            ],
            dim=-1,
        )
        visual_features = self.mlp(visual_features)
        return visual_features