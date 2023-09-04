import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
import jax.tree_util as jtu
from jaxtyping import Float, Array
import optax
from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.gaussian_hmm import ParamsSphericalGaussianHMMEmissions
from dynamax.deep_hidden_markov_model.models.abstractions import DeepHMM, DeepHMMEmissions
from dynamax.deep_hidden_markov_model.models.abstractions import DeepHMMParameterSet, DeepHMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.types import Scalar
from dynamax.utils.distributions import InverseWishart
from dynamax.utils.distributions import NormalInverseGamma
from dynamax.utils.distributions import NormalInverseWishart
from dynamax.utils.distributions import nig_posterior_update
from dynamax.utils.distributions import niw_posterior_update
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import pytree_sum
from typing import NamedTuple, Optional, Tuple, Union, List

import equinox as eqx

class _MLP(eqx.Module):
    """
    Class to construct an equinox module from a key and a eqx_list.
    """
    layers: list

    def __init__(self, key, eqx_list):
        """
        Parameters
        ----------
        key
            A jax random key
        eqx_list
            A list of list of successive equinox modules and activation functions to
            describe NN. The inner lists have the eqx module or
            axtivation function as first item, other items represents arguments
            that could be required (eg. the size of the layer).
            __Note:__ the `key` argument need not be given.
            Thus typical example is `eqx_list=
            [[eqx.nn.Linear, 2, 20],
                [jax.nn.tanh],
                [eqx.nn.Linear, 20, 20],
                [jax.nn.tanh],
                [eqx.nn.Linear, 20, 20],
                [jax.nn.tanh],
                [eqx.nn.Linear, 20, 1]
            ]`
        """

        self.layers = []
        # TODO we are limited currently in the number of layer type we can
        # parse and we lack some safety checks
        for l in eqx_list:
            if len(l) == 1:
                self.layers.append(l[0])
            else:
                # By default we append a random key at the end of the
                # arguments fed into a layer module call
                key, subkey = jr.split(key, 2)
                # the argument key is keyword only
                self.layers.append(l[0](*l[1:], key=subkey))

    def __call__(self, t):
        for layer in self.layers:
            t = layer(t)
        return t

def make_mlp(key, nn_architecture):
    mlp = _MLP(key, nn_architecture)
    params, static = eqx.partition(mlp, eqx.is_inexact_array)

    def init_fn():
        return params

    def apply_fn(emission, state, nn_params):
        model = eqx.combine(nn_params, static)
        ei = jnp.concatenate([emission.flatten(), state], axis=-1)
        return model(ei)

    return init_fn, apply_fn


class ParamsSphericalGaussianDeepHMMEmissions(NamedTuple):
    nn_params: Union[_MLP, ParameterProperties]
    means: Union[Float[Array, "num_timesteps state_dim emission_dim"],
            ParameterProperties, None]=None
    scales: Union[Float[Array, "num_timesteps state_dim"], ParameterProperties, None]=None

class SphericalGaussianDeepHMMEmissions(DeepHMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 nn_architecture,
                 num_lags=0,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.nn_architecture = nn_architecture
        self.num_lags = num_lags


    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize(self,
            key=jr.PRNGKey(0)):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """

        self.init_nn_params_fn, self.nn = make_mlp(key, self.nn_architecture)

        params = ParamsSphericalGaussianDeepHMMEmissions(
            nn_params=self.init_nn_params_fn(),
        )
        props = ParamsSphericalGaussianDeepHMMEmissions(
            nn_params=jtu.tree_map(lambda x:ParameterProperties(),
                params.nn_params), # NOTE that we construct a pytree of the
            # same shape as nn_params and no constrainer anywhere
            # this is ok since to_unconstrained and from_unconstrained operate
            # on Pytrees
            means=ParameterProperties(),
            scales=ParameterProperties(constrainer=tfb.Softplus())
        )
        return params, props

    def compute_means_and_covar(self, params, emissions, inputs=None):
        # Here, the function should be called compute_means_and_scales
        # Evaluate the NN on all the time steps with vmap
        f = lambda emission: \
            vmap(lambda state: self.nn(emission, state, params.nn_params))(
                    jnp.arange(self.num_states)[:, None]
            )
        if self.num_lags > 0:
            _emissions = jnp.stack([jnp.roll(emissions, shift=i, axis=0)
                for i in range(self.num_lags + 1)], axis=1)
            means_and_scales = vmap(f)(_emissions)
        else:
            means_and_scales = vmap(f)(emissions)
        # means_and_scales is shape (num_timesteps X num_states X 2 * emission_dim) (interlaced))        
        means = means_and_scales[..., ::self.num_states]
        scales = means_and_scales[..., 1::self.num_states]
        #jax.debug.print("{x}, {y}", x=jnp.amax(means), y=jnp.amin(means))
        #jax.debug.print("{x}, {y}", x=jnp.amax(scales), y=jnp.amin(scales))
        return (ParamsSphericalGaussianDeepHMMEmissions(
            nn_params=params.nn_params,
            means=means,
            scales=scales,
        ), ParamsSphericalGaussianHMMEmissions(
            means=means,
            scales=scales
        ))

    def distribution(self, params, state, inputs=None):
        dim = self.emission_dim
        return tfd.MultivariateNormalDiag(params.means[state],
                                          params.scales[state] * jnp.ones((dim,)))

    def log_prior(self, params):
        # We do not place a prior on the emission parameters
        return 0.0

class ParamsSphericalGaussianDeepHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsSphericalGaussianDeepHMMEmissions


class SphericalGaussianDeepHMM(DeepHMM):
    r"""An HMM with conditionally independent normal emissions with the same variance along
    each dimension. These are called *spherical* Gaussian emissions.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \prod_{n=1}^N \mathcal{N}(y_{t,n} \mid \mu_{z_t,n}, \sigma_{z_t}^2)$$
    or equivalently
    $$p(y_t \mid z_t, \theta) = \mathcal{N}(y_{t} \mid \mu_{z_t}, \sigma_{z_t}^2 I)$$

    where $\sigma_k^2$ is the *emission variance* in state $z_t=k$.
    The complete set of parameters is $\theta = \{\mu_k, \sigma_k^2\}_{k=1}^K$.

    The model has a non-conjugate, factored prior

    $$p(\theta) = \prod_{k=1}^K \mathcal{N}(\mu_{k} \mid \mu_0, \Sigma_0) \mathrm{Ga}(\sigma_{k}^2 \mid \alpha_0, \beta_0)$$

    *Note: In future versions we may implement a conjugate prior for this model.*

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param nn_architecture: the architecture for the NN
    :param m_step_optimizer: ``optax`` optimizer, like Adam.
    :param m_step_num_iters: number of optimizer steps per M-step.

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 nn_architecture: List,
                 num_lags: int=0,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 m_step_optimizer: optax.GradientTransformation=optax.adam(1e-2),
                 m_step_num_iters: int=50):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = SphericalGaussianDeepHMMEmissions(
            num_states, emission_dim,
            nn_architecture,
            num_lags,
            m_step_optimizer=m_step_optimizer,
            m_step_num_iters=m_step_num_iters)

        super().__init__(num_states, initial_component, transition_component, emission_component)


    @property
    def inputs_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's inputs.

        Needed since inputs are provided in SGD algorithms
        """
        return (1,)

    def initialize(self, key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                  transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
        ) -> Tuple[DeepHMMParameterSet, DeepHMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3)
        return ParamsSphericalGaussianDeepHMM(**params), ParamsSphericalGaussianDeepHMM(**props)
