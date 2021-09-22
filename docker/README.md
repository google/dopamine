Note: To use GPU, we expect an underlying CUDA version > 11.0.

We provide Dockerfiles for building docker images for Dopamine. We provide:

- A `core` [Dockerfile](core), which contains core libraries.
- An `atari` [Dockerfile](atari), which should be built on top of `core`
and provides setup for Atari environments.
- A `mujoco` [Dockerfile](mujoco), which should be built on top of `core`
and provides setup for Mujoco environments.

To start, build the core image, and then the image for the desired use-case
(i.e. `atari` or `mujoco`).

# Core Image

First, clone the repo:

```
git clone https://github.com/google/dopamine
```

Next, from the dopamine root (`cd dopamine`):

```
docker build -f docker/core/Dockerfile -t dopamine/core .
```

This will copy everything under the current directory into your image.

After the core image is built, you can build the [Atari image](#atari-image) or
the [Mujoco image](#mujoco-image). To build an image with both atari and
mujoco, see the [tips section](#tips).

# Atari Image

Once the [core image](#core-image) is built, you can build the Atari image.

To build the Atari image, you will also need a copy of the Atari roms
(Roms.rar), following the instructions from
[atari-py](https://github.com/openai/atari-py#roms). Assuming they are saved
in `$ROM_DIR`, you can run the following command to package them into an image:

```
docker build -f docker/atari/Dockerfile -t dopamine/atari $ROM_DIR
```

Once the image is built, see the [running images](#running-images) section.

# Mujoco Image

Once the [core image](#core-image) is built, you can build the Mujoco image.

To build the Mujoco image, you will need a mujoco key saved at
`~/.mujoco/mjkey.txt`, following the instructions on the
[Mujoco website](https://www.roboti.us/license.html). Once you have saved
your mujoco key, you can run the following command to build the image:

```
docker build -f docker/mujoco/Dockerfile -t dopamine/mujoco ~/.mujoco
```

Once the image is built, see the [running images](#running-images) section.

# Running Images

To run an image in interactive mode, use the following command:

```
docker run --gpus all --rm -it $IMAGE bash
```

replacing `$IMAGE` with `dopamine/atari` or `dopamine/mujoco` depending on
the image you built.

Once your image is built and you are able to run it, view the
[docs](https://google.github.io/dopamine/docs/) for next steps.

# Tips

## Building Combined Atari/Mujoco Image

To build an image that has Mujoco and Atari, follow the instructions
for building the Atari image. Then, follow the instructions for building
the mujoco image, but use the atari image as the base using the command
line flag `--build-arg base_image=dopamine/atari`.

## Using a Different CUDA Version

To use a different CUDA version, use the `cuda_docker_tag` arg when building
the core image
(e.g. `--build-arg cuda_docker_tag=11.1.1-cudnn8-devel-ubuntu20.04`).

See the [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) dockerhub page
for valid tags.

[core]: https://github.com/google/dopamine/blob/master/docker/core/Dockerfile
[atari]: https://github.com/google/dopamine/blob/master/docker/atari/Dockerfile
[mujoco]: https://github.com/google/dopamine/blob/master/docker/mujoco/Dockerfile
