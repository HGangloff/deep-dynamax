import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import vmap
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.deep_hidden_markov_model.models.abstractions import DeepHMMTransitions
from dynamax.parameters import ParameterProperties, from_unconstrained
from dynamax.utils.nn import _MLP, make_mlp, pretrain_nn
from jaxtyping import Float, Array
from typing import NamedTuple, Union

class ParamsDeepHMMTransitions(NamedTuple):
    nn_params: Union[_MLP, ParameterProperties]
    transition_matrix: Union[Float[Array, "state_dim state_dim"], ParameterProperties]=None


class DeepHMMTransitions(DeepHMMTransitions):
    r"""Deep model for HMM transitions.

    """
    def __init__(self, num_states, nn_architecture, concentration=1.1, stickiness=0.0):
        """
        Args:
            transition_matrix[j,k]: prob(hidden(t) = k | hidden(t-1)j)
        """
        self.num_states = num_states
        self.nn_architecture = nn_architecture
        self.concentration = \
            concentration * jnp.ones((num_states, num_states)) + \
                stickiness * jnp.eye(num_states)

    def distribution(self, params, state, inputs=None):
        return tfd.Categorical(probs=params.transition_matrix[state])

    def initialize(self, key=jr.PRNGKey(0), method="prior",
            transition_matrix=None, pretrain_transitions=None):
        """Initialize the model parameters and their corresponding properties.

        Args:
            key (_type_, optional): _description_. 
            method (str, optional): _description_. Defaults to "prior".
            transition_matrix (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        self.init_nn_params_fn, self.nn = make_mlp(key, self.nn_architecture)

        if transition_matrix is None:
            this_key, key = jr.split(key)
            transition_matrix = tfd.Dirichlet(self.concentration).sample(seed=this_key)

        # Package the results into dictionaries
        params = ParamsDeepHMMTransitions(
            nn_params=self.init_nn_params_fn(),
        )

        props = ParamsDeepHMMTransitions(
            nn_params=jtu.tree_map(lambda x:ParameterProperties(),
                params.nn_params),
            transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())
        )

        if pretrain_transitions is not None:
            params = params._replace(
                nn_params=pretrain_nn(
                    self.nn,
                    params.nn_params,
                    props,
                    pretrain_transitions
                )
            )
        return params, props

    def log_prior(self, params):
        return 0.0

    def _compute_transitions_nn(self, params, props, emissions=None, inputs=None):
        """As opposed to classical HMMTransitions, we here have the previous
        emission at our disposal at each time step
        """
        f = lambda emission: \
            vmap(lambda state: self.nn(emission, state, params.nn_params))(
                jnp.arange(self.num_states)[:, None]
            )
        # raw transitions at the nn output with shape (num_timesteps X
        # num_states X num_states)
        transitions_nn = vmap(f)(jnp.roll(emissions, shift=1, axis=0))
        transitions_nn = from_unconstrained(
            transitions_nn,
            props.transition_matrix
        )
        return ParamsDeepHMMTransitions(
            nn_params=params.nn_params,
            transition_matrix=transitions_nn
        ), ParamsStandardHMMTransitions(
            transition_matrix=transitions_nn
        )
        #return params.transition_matrix

    #def collect_suff_stats(self, params, posterior, inputs=None):
    #    return posterior.trans_probs

    #def initialize_m_step_state(self, params, props):
    #    return None

    #def m_step(self, params, props, batch_stats, m_step_state):
    #    if props.transition_matrix.trainable:
    #        if self.num_states == 1:
    #            transition_matrix = jnp.array([[1.0]])
    #        else:
    #            expected_trans_counts = batch_stats.sum(axis=0)
    #            transition_matrix = tfd.Dirichlet(self.concentration + expected_trans_counts).mode()
    #        params = params._replace(transition_matrix=transition_matrix)
    #    return params, m_step_state
