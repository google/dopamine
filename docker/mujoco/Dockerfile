ARG base_image=dopamine/core
FROM ${base_image}

# Create Mujoco subdir.
RUN mkdir /root/.mujoco
COPY mjkey.txt /root/.mujoco/mjkey.txt

# Prerequisites
RUN apt-get install \
  libosmesa6-dev \
  libgl1-mesa-glx \
  libglfw3 \
  libglew-dev \
  patchelf \
  gcc \
  python3.8-dev \
  unzip -y

# Download and install mujoco.
RUN wget https://www.roboti.us/download/mujoco200_linux.zip
RUN unzip mujoco200_linux.zip
RUN rm mujoco200_linux.zip
RUN mv mujoco200_linux /root/.mujoco/mujoco200

# Add LD_LIBRARY_PATH environment variable.
ENV LD_LIBRARY_PATH "/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}"
RUN echo 'export LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}' >> /etc/bash.bashrc

# Finally, install mujoco_py.
RUN pip install mujoco_py
