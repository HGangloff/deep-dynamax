import jax.random as jr
import jax.numpy as jnp
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
