import jax.numpy as jnp

from ssm_jax.unscented_kalman_filter.inference import unscented_kalman_smoother, UKFHyperParams
from ssm_jax.nonlinear_gaussian_ssm.sarkka_lib import ukf, uks
from ssm_jax.nonlinear_gaussian_ssm.inference_test import random_nlgssm_args


# Helper functions
_all_close = lambda x, y: jnp.allclose(x, y, atol=1e-4)


def test_ukf_nonlinear(key=0, num_timesteps=15):
    nlgssm_args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)
    hyperparams = UKFHyperParams()

    # Run UKF from sarkka-jax library
    means_ukf, covs_ukf = ukf(*(nlgssm_args.values()), *(hyperparams.to_tuple()), emissions)
    # Run UKS from sarkka-jax library
    means_uks, covs_uks = uks(*(nlgssm_args.values()), *(hyperparams.to_tuple()), emissions)
    # Run UKS from SSM-Jax
    uks_post = unscented_kalman_smoother(nlgssm_args, emissions, hyperparams)

    # Compare filter results
    assert _all_close(means_ukf, uks_post.filtered_means)
    assert _all_close(covs_ukf, uks_post.filtered_covariances)
    assert _all_close(means_uks, uks_post.smoothed_means)
    assert _all_close(covs_uks, uks_post.smoothed_covariances)