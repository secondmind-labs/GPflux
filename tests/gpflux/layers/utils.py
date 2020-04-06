import gpflow
import gpflux


def build_gp_layer(
    num_data: int, num_inducing: int, input_dim: int, output_dim: int
) -> gpflux.layers.GPLayer:
    """
    Builds a vanilla GP layer
    """

    base_kernel = gpflow.kernels.RBF(lengthscales=[1.0] * input_dim)
    kernel = gpflux.helpers.construct_basic_kernel(
        base_kernel, output_dim=output_dim, share_hyperparams=True,
    )
    inducing_variable = gpflux.helpers.construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim=output_dim, share_variables=True,
    )
    gp_layer = gpflux.layers.GPLayer(
        kernel=kernel,
        inducing_variable=inducing_variable,
        num_data=num_data,
        mean_function=gpflow.mean_functions.Zero(),
    )
    return gp_layer
