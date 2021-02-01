# ConvSegFault

Minimal example of using XLA within the Erlang Runtime System. Attempting
to run convolutions on a dirty scheduler thread produces a segmentation
fault.

## Running

```shell
docker build --rm -t conv:cuda11.1 .
```

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -e CONV_CACHE=$PWD/tmp/conv_cache \
  -w $PWD \
  --gpus=all \
  --rm conv:cuda11.1 bash
```

Inside the docker container:

```shell
$ mix run scripts/conv.exs
```

The above script will run successfully and return `:ok`.

```shell
$ mix run scripts/seg_fault.exs
```

The above script will SegFault. There is no difference between the functions called in either script, aside from the fact that 1 is scheduled on an Erlang dirty I/O scheduler (see [Dirty NIF](https://erlang.org/doc/man/erl_nif.html)).

You can play around with the functions like this:

```shell
$ iex -S mix

iex> ConvSegFault.conv()
...logs...
:ok
iex> ConvSegFault.conv_seg_fault()
:ok
```

It seems that running the function that always succeeds first, causes the function that sometimes SegFaults to always succeed. If you tried this:

```shell
$ iex -S mix

iex> ConvSegFault.conv_seg_fault()
...logs...
Segmentation Fault (core dumped)
```

it would always fail.