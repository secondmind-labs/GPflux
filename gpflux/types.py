# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


# Ideally this is Union[tf.Variable, tf.Tensor, np.ndarray]
# but that doesn't play nicely with Multiple Dispatch
TensorLike = object
