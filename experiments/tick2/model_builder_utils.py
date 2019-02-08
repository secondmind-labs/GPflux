import numpy as np
import tensorflow as tf
from sklearn import cluster

import gpflow
import gpflux
from gpflow.conditionals import Kuu


def cluster_patches(NHWC_X, M, patch_size):
    # from https://github.com/kekeblom/DeepCGP/blob/master/conv_gp/kernels.py 
    # 09/01/2019
        def _sample(tensor, count):
            chosen_indices = np.random.choice(np.arange(tensor.shape[0]), count)
            return tensor[chosen_indices]

        def _sample_patches(HW_image, N, patch_size, patch_length):
            out = np.zeros((N, patch_length))
            for i in range(N):
                patch_y = np.random.randint(0, HW_image.shape[0] - patch_size)
                patch_x = np.random.randint(0, HW_image.shape[1] - patch_size)
                out[i] = HW_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size].reshape(patch_length)
            return out

        NHWC = NHWC_X.shape
        patch_length = patch_size ** 2 * NHWC[3]
        # Randomly sample images and patches.
        patches = np.zeros((M, patch_length), dtype=gpflow.settings.float_type)
        patches_per_image = 1
        samples_per_inducing_point = 100
        for i in range(M * samples_per_inducing_point // patches_per_image):
            # Sample a random image, compute the patches and sample some random patches.
            image = _sample(NHWC_X, 1)[0]
            sampled_patches = _sample_patches(image, patches_per_image,
                    patch_size, patch_length)
            patches[i*patches_per_image:(i+1)*patches_per_image] = sampled_patches

        k_means = cluster.KMeans(n_clusters=M, init='random', n_jobs=-1)
        k_means.fit(patches)
        return k_means.cluster_centers_


def build_hidden_layer(
        X_cluster_init,
        input_image_shape,
        feature_maps_out,
        patch_shape,
        num_inducing_points):

    assert len(input_image_shape) == 3  # [W, H, C]

    patches1 = cluster_patches(X_cluster_init, num_inducing_points, patch_shape[0])
    feat1 = gpflux.convolution.InducingPatch(patches1)
    base_kern1 = gpflow.kernels.RBF(np.prod(patch_shape) * input_image_shape[-1])
    base_kern1.lengthscales = 1.5
    kern1 = gpflux.convolution.ConvKernel(
        base_kern1,
        image_shape=input_image_shape,
        patch_shape=patch_shape,
        pooling=1,
        with_indexing=False
    )

    q_mu1 = np.zeros((num_inducing_points, feature_maps_out))
    q_sqrt1 = np.tile(np.eye(num_inducing_points)[None, :, :], [feature_maps_out, 1, 1]) * 1e-5

    mean1 = gpflux.convolution.IdentityConvMean(
        image_shape=input_image_shape,
        filter_shape=patch_shape,
        feature_maps_out=feature_maps_out
    )

    layer = gpflux.layers.GPLayer(
        kern1,
        feat1,
        num_latents=feature_maps_out,
        q_mu=q_mu1,
        q_sqrt=q_sqrt1,
        mean_function=mean1,
    )

    output_image_shape = [kern1.config.Hout, kern1.config.Wout, feature_maps_out]
    output_images = mean1.compute_default_graph(X_cluster_init[:100])
    output_images = output_images.reshape([-1, *output_image_shape])

    return layer, output_image_shape, output_images


def build_final_layer(
        X_cluster_init,
        input_image_shape,
        num_classes,
        patch_shape,
        num_inducing_points,
        tick,
        weights):

    patches2 = cluster_patches(X_cluster_init, num_inducing_points, patch_shape[0])
    if tick:
        indices = np.random.randint(0, input_image_shape[0], size=[num_inducing_points, 2])
        indices2 = indices.astype(np.float64) / indices.max(axis=0)
        feat2 = gpflux.convolution.IndexedInducingPatch(patches2, indices2)
    else:
        feat2 = gpflux.convolution.InducingPatch(patches2)
    base_kern2 = gpflow.kernels.RBF(np.prod(patch_shape)*input_image_shape[-1])
    kern2 = gpflux.convolution.WeightedSumConvKernel(
        base_kern2,
        image_shape=input_image_shape,
        patch_shape=patch_shape,
        pooling=1, 
        with_indexing=tick,
        with_weights=weights
    )

    num_latents = 1 if num_classes == 2 else num_classes
    q_mu2 = np.zeros((num_inducing_points, num_latents))
    q_sqrt2 = np.tile(np.eye(num_inducing_points)[None, :, :], [num_latents, 1, 1])

    layer = gpflux.layers.GPLayer(
        kern2,
        feat2,
        num_latents=num_latents,
        q_mu=q_mu2,
        q_sqrt=q_sqrt2,
        mean_function=None,
    )

    return layer


def save_model_parameters(model, filename):
    model.anchor(model.enquire_session())
    params = {}
    for param in model.parameters:
        value = param.read_value()
        key = param.pathname
        params[key] = value
    np.save(filename, params)


def number_of_layers_in_saved_parameters(filename):
    parameters = np.load(filename).item()
    layer_numbers = [int(k.split('/')[2]) for k in parameters]
    return max(layer_numbers) + 1


def load_model_parameters(layers, filename):
    """ 
    `growing` means that we are using these params in a model with
    an extra layer. We can not set the weights in that case.
    """
    num_layers_in_init = number_of_layers_in_saved_parameters(filename)
    if num_layers_in_init == len(layers):
        growing = False
    elif num_layers_in_init + 1 == len(layers):
        # init all layers except the before last one
        layers = layers[:-2] + layers[-1:]
        growing = True
    else:
        raise ValueError("Cannot load model parameters, wrong number of layers")

    parameters = np.load(filename).item()

    def parse_layer_path(key):
        if 'layers' not in key:
            return None, None
        parts = key.split('/')
        return int(parts[2]), "/".join(parts[3:])

    for key, value in parameters.items():
        layer, path = parse_layer_path(key)
        if layer is None:
            continue

        if 'q_mu' in path:
            layers[layer].q_mu = value
        elif 'q_sqrt' in path:
            layers[layer].q_sqrt = value
        elif 'feature/Z' in path:
            layers[layer].feature.Z = value
        elif 'feature/indices' in path:
            layers[layer].feature.indices = value
        elif 'basekern/variance' in path:
            layers[layer].kern.basekern.variance = value
        elif 'basekern/lengthscales' in path:
            layers[layer].kern.basekern.lengthscales = value
        elif 'index_kernel/variance' in path:
            layers[layer].kern.index_kernel.variance = value
        elif 'index_kernel/lengthscales' in path:
            layers[layer].kern.index_kernel.lengthscales = value
        elif 'weights' in path:
            if not growing:
                layers[layer].kern.weights = value
        else:
            raise ValueError(f"Not able to assign value for {path}")
    
    return layers

def build_model(
    X,
    Y,
    *,
    num_layers=None,
    feature_maps_out=None,
    patch_shape=None,
    num_inducing_points=None,
    tick=None,
    weights=None,
    batch_size=32,
    likelihood=None,
    init_file=None):

    assert likelihood in ["soft", "robust", "bern"]
    assert len(X.shape) == 4  # [N, 1]
    assert len(Y.shape) == 2  # [N, H, W, C]
    assert len(feature_maps_out) == num_layers - 1
    assert num_layers in [1, 2, 3]
    assert len(patch_shape) == 2  # [h, w]

    with gpflow.defer_build():
        input_image_shape = X.shape[1:]
        num_classes = Y.max() - Y.min() + 1
        imgs = X

        layers = []
        for i in range(num_layers - 1):
            layer, input_image_shape, imgs = build_hidden_layer(
                X_cluster_init=imgs, 
                input_image_shape=input_image_shape,
                feature_maps_out=feature_maps_out[i],  # keep num features == num color channels
                patch_shape=patch_shape,
                num_inducing_points=num_inducing_points,
            )
            layers.append(layer)

        layer = build_final_layer(
            X_cluster_init=imgs,
            input_image_shape=input_image_shape,
            num_classes=num_classes if num_classes > 2 else 1,
            patch_shape=patch_shape,
            num_inducing_points=num_inducing_points,
            tick=tick,
            weights=weights,
        )
        layer.kern.basekern.variance = 15.0
        layer.kern.basekern.lengthscales = 15.0
        if tick:
            layer.kern.index_kernel.variance = 5.0
            layer.kern.index_kernel.lengthscales = 5.0

        layers.append(layer)

        if init_file:
            load_model_parameters(layers, init_file)

        if likelihood == "soft":
            like = gpflow.likelihoods.SoftMax(num_classes)
        elif likelihood == "robust":
            like = gpflow.likelihoods.MultiClass(num_classes)
        elif num_classes == 2 and likelihood == "bern":
            like = gpflow.likelihoods.Bernoulli()
        else:
            raise ValueError("Invalid likelihood")

        model = gpflux.DeepGP(
            X.reshape(X.shape[0], -1),
            Y,
            layers=layers,
            likelihood=like,
            batch_size=batch_size
        )

    model.compile()

    return model



if __name__ == "__main__":
    N, H, W, C = 100, 32, 32, 3
    num_classes = 10
    M = 384
    X = np.random.randn(N, H, W, C)
    Y = np.random.randint(0, num_classes, size=(N, 1))
    patch_shape = h, w = [5, 5]

    filename = "./params2.npy"

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(model.compute_log_likelihood())

    model = build_model(
        X,
        Y,
        num_layers=2,
        feature_maps_out=[3],
        patch_shape=patch_shape,
        num_inducing_points=M,
        tick=False,
        init_file=None
    )
    print(model)
    print(model.compute_log_likelihood())
    save_model_parameters(model, filename)


    model = build_model(
        X,
        Y,
        num_layers=3,
        feature_maps_out=[3, 3],
        patch_shape=patch_shape,
        num_inducing_points=M,
        tick=True,
        init_file=filename
    )
    print("AFTER")
    print(model)
    print(model.compute_log_likelihood())
