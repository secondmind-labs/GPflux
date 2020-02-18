notebooks = conditional_deep_gp.ipynb deep_nonstationary_gp_samples.ipynb

.SUFFIXES: .ipynb .py

.PHONY: all
all: $(notebooks)

$(notebooks): %.ipynb: %.py
	jupytext --execute --to notebook $<
