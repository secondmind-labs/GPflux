bash run_tickconvgp_fractions_grey_cifar10.sh "$1"
wait
bash run_tickconvgp_mixed_mnist.sh "$1"
wait
bash run_tickconvgp_fractions_svhn.sh "$1"