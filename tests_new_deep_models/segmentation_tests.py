import os
from os import listdir
from os.path import isfile, join

os.environ['CUDA_VISIBLE_DEVICES'] = '' # uncomment to force CPU

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from jax import vmap
import equinox as eqx
import numpy as np
from PIL import Image

from dynamax.hidden_markov_model import SphericalGaussianHMM 
from dynamax.deep_hidden_markov_model import SphericalGaussianDeepHMM 
from dynamax.parameters import to_unconstrained, from_unconstrained

from chain_to_image_functions import chain_to_image, image_to_chain

img_dir = "./cattles256/"
images = [img for img in listdir(img_dir) if isfile(join(img_dir, img))]

try:
    H = np.load("image_to_chain.npy")
except FileNotFoundError:
    img = np.array(Image.open(join(img_dir, images[0])))
    img[img == 255] = 1
    #H = image_to_chain(img)[:]
    H = img.flatten()
    np.save("image_to_chain.npy", H)
obs = [0.5]
b = jnp.array([0., 0.5])
for i in range(1, 256*256):
    obs.append(np.sin(b[H[i]] + obs[-1]) + np.random.randn()*0.5)
    #obs.append(H[i] + np.random.randn()*0.5)
X = jnp.array(obs)[:, None]
plt.imshow(X.reshape((256, 256)))
plt.show()
T = len(H)

nb_classes = 2 # num_states
nb_channels = 1 # emission dims
num_lags = 1

# EM with non deep model
hmm = SphericalGaussianHMM(nb_classes, nb_channels)
em_params, em_props = hmm.initialize(jr.PRNGKey(0))
em_params, log_probs = hmm.fit_em(
    em_params,
    em_props,
    X,
    num_iters=100
)
em_loss = -log_probs / X.size

em_most_likely_states = hmm.most_likely_states(
    em_params,
    em_props,
    X
)

nn_architecture_emissions = [
    [eqx.nn.Linear, 1 + num_lags, 100],
    [jax.nn.relu],
    #[eqx.nn.Linear, 20, 20],
    #[jax.nn.relu],
    [eqx.nn.Linear, 100, 2 * nb_channels]
        ]

nn_architecture_transitions = [
    [eqx.nn.Linear, 2, 100],
    [jax.nn.relu],
    #[eqx.nn.Linear, 20, 20],
    #[jax.nn.relu],
    [eqx.nn.Linear, 100, nb_classes - 1]
        ]

unc_em_params = to_unconstrained(em_params, em_props)

pretrain_transitions = {
    "inputs":jnp.concatenate([
        jnp.roll(X, shift=1, axis=0),
        em_most_likely_states[:, None]
        ],
        axis=1
    )[..., None],
    "outputs":jnp.repeat(
        unc_em_params.transitions.transition_matrix[None],
        X.shape[0],
        axis=0
    ),
    "optimizer":optax.adam(1e-2),
    "optimizer_state":None,
    "num_mstep_iters":50
}

pretrain_emissions = {
    "inputs":jnp.concatenate([
        jnp.roll(X, shift=1, axis=0),
        em_most_likely_states[:, None]
        ],
        axis=1
    )[..., None],
    "outputs":jnp.repeat(
        jnp.concatenate([
            unc_em_params.emissions.means, 
            unc_em_params.emissions.scales[:, None]
            ], 
            axis=0
        ).reshape((2, -1), order='F')[None],
        X.shape[0],
        axis=0
    ),
    "optimizer":optax.adam(1e-2),
    "optimizer_state":None,
    "num_mstep_iters":50
}

dhmm = SphericalGaussianDeepHMM(
    nb_classes,
    nb_channels,
    nn_architecture_emissions=nn_architecture_emissions,
    nn_architecture_transitions=nn_architecture_transitions,
    num_lags=num_lags
)

dhmm_params, dhmm_props = dhmm.initialize(
    jr.PRNGKey(0),
    pretrain_emissions=pretrain_emissions,
    pretrain_transitions=pretrain_transitions
)

dhmm_params = dhmm_params._replace(
    emissions=dhmm.emission_component.compute_means_and_covar_nn(
        dhmm_params.emissions,
        dhmm_props.emissions,
        X
    )[0],
    transitions=dhmm.transition_component._compute_transitions_nn(
        dhmm_params.transitions,
        dhmm_props.transitions,
        X,
    )[0]
)

dhmm_most_likely_states = dhmm.most_likely_states(
    dhmm_params,
    dhmm_props,
    X
)

dhmm_params, dhmm_losses = dhmm.fit_sgd(
    dhmm_params,
    dhmm_props,
    X,
    optimizer=optax.adam(1e-3),
    batch_size=256,
    num_epochs=100
)

dhmm_most_likely_states = dhmm.most_likely_states(
    dhmm_params,
    dhmm_props,
    X
)

plt.plot(dhmm_losses)
plt.plot(em_loss)
plt.show()

fig, axes = plt.subplots(1, 2)
axes[0].imshow(em_most_likely_states.reshape((256, 256)))
axes[1].imshow(dhmm_most_likely_states.reshape((256, 256)))
plt.show()

hmm = SphericalGaussianHMM(nb_classes, nb_channels)

models = [hmm, hmm, dhmm]
lrs = [1e-3, 1e-3, 1e-3]
methods = ["em", "sgd", "sgd"]

losses_fig, losses_axes = plt.subplots(1, 1)
viterbi_fig, viterbi_axes = plt.subplots(1, len(models))

for idx, (model, lr, method) in enumerate(zip(models, lrs, methods)):
    params, props = model.initialize(jr.PRNGKey(0))
    if method == "sgd":
        params, losses = model.fit_sgd(
            params,
            props,
            X,
            optimizer=optax.adam(lr),
            batch_size=256,
            num_epochs=500
        )
    else:
        params, log_probs = model.fit_em(
            params,
            props,
            X,
            num_iters=100
        )
        losses = -log_probs / X.size

    if idx == 2:
        print(params)
        print("--")
        print(
                model.transition_component._compute_transitions_nn(
                    params.transitions,
                    props.transitions,
                    X
                )[0],
        )
        print("--")
        print(
            model.emission_component.compute_means_and_covar_nn(params.emissions,
                props.emissions, X)[0]
        )

    most_likely_states = model.most_likely_states(params, props, X, inputs=H[:, None])
    #plot_map_sequence(most_likely_states[:num_timesteps], H[:num_timesteps])
    print(most_likely_states.shape)
    viterbi_axes[idx].imshow(most_likely_states.reshape((256, 256)))
    losses_axes.plot(losses)

plt.show()

#num_timesteps = 10000
#def plot_posterior_probs(probs, states, title=""):
#    plt.imshow(states[None, :], extent=(0, num_timesteps, 0, 1), 
#               interpolation="none", aspect="auto", cmap="Greys", alpha=0.25)
#    plt.plot(probs[:, 1])   # probability of the loaded state (z=1)
#    plt.xlabel("time")
#    plt.ylabel("p(loaded)")
#    plt.ylim(0, 1)
#    plt.title(title)
#    plt.show()
#
#posterior = hmm.smoother(params, X)
#plot_posterior_probs(posterior.smoothed_probs[:num_timesteps], H[:num_timesteps], title="Smoothing distribution")
#
#def plot_map_sequence(most_likely_states, states):
#    plt.imshow(states[None, :], extent=(0, num_timesteps, -0.05, 1.05), 
#               interpolation="none", aspect="auto", cmap="Greys", alpha=0.25)
#    plt.plot(most_likely_states)
#    plt.xlabel("time")
#    plt.ylabel("MAP state")
#    plt.ylim(-0.05, 1.05)
#    plt.yticks([0, 1])
#    plt.title("Viterbi estimate")
#    plt.show()

