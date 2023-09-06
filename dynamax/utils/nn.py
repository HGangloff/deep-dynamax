import jax
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
import equinox as eqx

from dynamax.utils.utils import ensure_array_has_batch_dim
from dynamax.utils.optimize import run_gradient_descent

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

def pretrain_nn(nn, nn_params, props, pretrain_params):
    """
    pretrain_params (dict) with keys inputs, outputs, optimizer,
    optimizer_state, num_mstep_iters
    """
    # the output (ie params) should be given in their unconstrained form!!!
    # unc_params = to_unconstrained(params, props)

    #batch_outputs = ensure_array_has_batch_dim(
    #    pretrain_params["outputs"],
    #    pretrain_params["outputs"]
    #)
    #batch_inputs = ensure_array_has_batch_dim(
    #    pretrain_params["inputs"],
    #    pretrain_params["inputs"]
    #)
    def mse(nn_params):
        vloss = vmap(lambda inpt, output: nn(inpt[0], inpt[1], nn_params) -
                output[inpt[1].astype(int)])
            pretrain_params["inputs"][0, 1], nn_params))
        return jnp.mean(
            vloss(
                pretrain_params["inputs"],
                pretrain_params["outputs"]
            ) ** 2
        )

    # Run gradient descent
    nn_params, m_step_state, losses = run_gradient_descent(
        mse,
        nn_params,
        optimizer=pretrain_params["optimizer"],
        optimizer_state=pretrain_params["optimizer_state"],
        num_mstep_iters=pretrain_params["num_mstep_iters"]
    )
    return nn_params
