import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import io


def get_classification_detailed_plot(
    num_layers, X_training, Y_training, where_to_save, f_mean_overall, f_var_overall, name_file
):

    xx, yy = np.mgrid[-5:5:0.1, -5:5:0.1]
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = grid.astype(np.float32)

    indices_class_1 = np.where(Y_training == 1.0)
    indices_class_0 = np.where(Y_training == 0.0)

    fig, axs = plt.subplots(
        nrows=2, ncols=num_layers, sharex=True, sharey=True, figsize=(20 * num_layers, 40)
    )

    for current_layer in range(num_layers):

        current_mean = f_mean_overall[current_layer]
        current_mean = current_mean.reshape((100, 100))
        current_var = f_var_overall[current_layer]
        current_var = current_var.reshape((100, 100))

        ###################
        ##### F mean  #####
        ###################

        axis = axs[0, current_layer]
        contour = axis.contourf(xx, yy, current_mean, 50, cmap="coolwarm")
        cbar1 = fig.colorbar(contour, ax=axis)

        cbar1.ax.tick_params(labelsize=60)

        axis.set(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), xlabel="$X_1$", ylabel="$X_2$")
        axis.set_title(label="Predictive Mean", fontdict={"fontsize": 60})
        axis.tick_params(axis="both", which="major", labelsize=80)

        axis.scatter(
            X_training[indices_class_0, 0],
            X_training[indices_class_0, 1],
            s=100,
            marker="X",
            alpha=0.2,
            c="green",
            linewidth=1,
            label="Class 0",
        )
        axis.scatter(
            X_training[indices_class_1, 0],
            X_training[indices_class_1, 1],
            s=100,
            marker="D",
            alpha=0.2,
            c="purple",
            linewidth=1,
            label="Class 1",
        )

        # TODO -- don't need this
        # axis.scatter(Z_np[current_layer][:,0], Z_np[current_layer][:,1],
        #        s=750, marker="*", alpha=0.95, c = 'cyan',
        #        linewidth=1, label = 'Inducing Points')

        axis.legend(loc="upper right", prop={"size": 60})
        # axis.text(-4.5, 4.5, 'LL:'+"{:.2f}".format(total_nll_np)+'; Acc:'+"{:.2f}".format(precision_testing_overall_np), size=50, color='black')

        #################################
        ##### F var Distributional  #####
        #################################

        axis = axs[1, current_layer]
        contour = axis.contourf(xx, yy, current_var, 50, cmap="coolwarm")
        cbar1 = fig.colorbar(contour, ax=axis)
        cbar1.ax.tick_params(labelsize=60)

        axis.set(xlim=(-5, 5), ylim=(-5, 5), xlabel="$X_1$", ylabel="$X_2$")

        axis.set_title(label="Predictive Variance", fontdict={"fontsize": 60})
        axis.tick_params(axis="both", which="major", labelsize=80)

        axis.scatter(
            X_training[indices_class_0, 0],
            X_training[indices_class_0, 1],
            s=100,
            marker="X",
            alpha=0.2,
            c="green",
            linewidth=1,
            label="Class 0",
        )
        axis.scatter(
            X_training[indices_class_1, 0],
            X_training[indices_class_1, 1],
            s=100,
            marker="D",
            alpha=0.2,
            c="purple",
            linewidth=1,
            label="Class 1",
        )

        # axis.scatter(Z_np[current_layer][:,0], Z_np[current_layer][:,1],
        #    s=750, marker="*", alpha=0.95, c = 'cyan',
        #    linewidth=1, label = 'Inducing Points')
        axis.legend(loc="upper right", prop={"size": 60})

    plt.tight_layout()
    plt.savefig(where_to_save + name_file)
    plt.close()


def get_regression_detailed_plot(
    num_layers,
    X_training,
    Y_training,
    where_to_save,
    mean,
    var,
    name_file,
    x_margin,
    y_margin,
    X_test,
):

    figure, axs = plt.subplots(
        nrows=1, ncols=num_layers, sharex=True, sharey=True, figsize=(10 * num_layers, 10)
    )

    for current_layer in range(num_layers):
        current_mean = mean[current_layer]
        current_var = var[current_layer]

        ###################
        ##### F mean  #####
        ###################

        X_test = X_test.squeeze()
        lower = current_mean - 2 * np.sqrt(current_var)
        upper = current_mean + 2 * np.sqrt(current_var)

        axis = axs[current_layer]
        axis.set_ylim(Y_training.min() - y_margin, Y_training.max() + y_margin)
        axis.plot(X_training, Y_training, "kx", alpha=0.5, label="Training")
        axis.plot(X_test, current_mean, "C1")

        axis.fill_between(X_test, lower, upper, color="C1", alpha=0.3)
        axis.legend(loc="upper right", prop={"size": 60})

        axis.set_title(label=f"Layer {current_layer+1}", fontdict={"fontsize": 60})
        axis.tick_params(axis="both", which="major", labelsize=80)

    plt.tight_layout()
    # plt.savefig(where_to_save+name_file)
    # plt.close()
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
