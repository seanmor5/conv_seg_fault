# System vars
TEMP ?= $(HOME)/.cache

# mix.exs vars
# ERTS_INCLUDE_DIR

# Public configuration
CONV_MODE ?= opt # can also be dbg
CONV_CACHE ?= $(TEMP)/conv
CONV_TENSORFLOW_GIT_REPO ?= https://github.com/tensorflow/tensorflow.git
CONV_TENSORFLOW_GIT_REV ?= 458a464a3c8e4a5a64186ff17f6982e30de2a030

# Private configuration
CONV_SO = priv/libconv.so
CONV_DIR = c_src/conv
ERTS_SYM_DIR = $(CONV_DIR)/erts
BAZEL_FLAGS = --define "framework_shared_object=false" -c $(CONV_MODE)

TENSORFLOW_NS = tf-$(CONV_TENSORFLOW_GIT_REV)
TENSORFLOW_DIR = $(CONV_CACHE)/$(TENSORFLOW_NS)
TENSORFLOW_CONV_NS = tensorflow/compiler/xla/conv
TENSORFLOW_CONV_DIR = $(TENSORFLOW_DIR)/$(TENSORFLOW_CONV_NS)

all: symlinks
	cd $(TENSORFLOW_DIR) && \
		bazel build $(BAZEL_FLAGS) --config=cuda //$(TENSORFLOW_CONV_NS):libconv.so
	mkdir -p priv
	cp -f $(TENSORFLOW_DIR)/bazel-bin/$(TENSORFLOW_CONV_NS)/libconv.so $(CONV_SO)

symlinks: $(TENSORFLOW_DIR)
	rm -f $(TENSORFLOW_CONV_DIR)
	ln -s "$(MIX_CURRENT_PATH)/$(CONV_DIR)" $(TENSORFLOW_CONV_DIR)
	rm -f $(ERTS_SYM_DIR)
	ln -s "$(ERTS_INCLUDE_DIR)" $(ERTS_SYM_DIR)

# Print Tensorflow Dir
PTD:
	@ echo $(TENSORFLOW_DIR)

# Clones tensorflow
$(TENSORFLOW_DIR):
	mkdir -p $(TENSORFLOW_DIR)
	cd $(TENSORFLOW_DIR) && \
		git init && \
		git remote add origin $(CONV_TENSORFLOW_GIT_REPO) && \
		git fetch --depth 1 origin $(CONV_TENSORFLOW_GIT_REV) && \
		git checkout FETCH_HEAD

clean:
	cd $(TENSORFLOW_DIR) && bazel clean --expunge
	rm -f $(ERTS_SYM_DIR) $(TENSORFLOW_CONV_DIR)
	rm -rf $(CONV_SO) $(TENSORFLOW_DIR)
