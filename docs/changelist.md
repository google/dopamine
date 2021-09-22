# Changelist

********************************************************************************

Note: This changelist attempts to summarize major changes to the Dopamine
library. For a more fine-grained changelist, consider looking through the
[commit history](https://github.com/google/dopamine/commits/master).

*   **21/09/2021:** Added Dockerfiles and instructions for using Dopamine with
    docker.

*   **07/09/2021:** Migrated JAX agents to use
    [Optax](https://github.com/deepmind/optax) to create optimizers, as
    `flax.optim` is
    [being deprecated](https://flax.readthedocs.io/en/latest/flax.optim.html)

*   **25/08/2021:** Added SAC and continuous control training library. Added
    Atari 100k to Dopamine labs.

*   **29/06/2021:** Added full Rainbow. Full Rainbow includes double DQN, noisy
    networks, and dueling DQN, on top of the components in our earlier
    implementation (n-step updates, prioritized replay, and distirbutional RL).

*   **03/03/2021:** Updated flax networks to flax.linen.

*   **16/10/2020:** Learning curves for the
    [QR-DQN JAX agent](https://github.com/google/dopamine/blob/master/dopamine/jax/agents/quantile/quantile_agent.py)
    have been added to the
    [baseline plots](https://google.github.io/dopamine/baselines/plots.html)!

*   **03/08/2020:** Dopamine now supports [JAX](https://github.com/google/jax)
    agents! This includes an implementation of the Quantile Regression agent
    (QR-DQN) which has been a common request. Find out more in our
    [jax](https://github.com/google/dopamine/tree/master/dopamine/jax)
    subdirectory, which includes trained agent checkpoints.

*   **27/07/2020:** Dopamine now runs on TensorFlow 2. However, Dopamine is
    still written as TensorFlow 1.X code. This means your project may need to
    explicity disable TensorFlow 2 behaviours with:

    ```
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    ```

    if you are using custom entry-point for training your agent. The migration
    to TensorFlow 2 also means that Dopamine no longer supports Python 2.

*   **02/09/2019:** Dopamine has switched its network definitions to use
    tf.keras.Model. The previous `tf.contrib.slim` based networks are removed.
    If your agents inherit from dopamine agents you need to update your code.

    *   `._get_network_type()` and `._network_template()` functions are replaced
        with `._create_network()` and `network_type` definitions are moved
        inside the model definition.

        ```
        # The following two functions are replaced with `_create_network()`.
        # def _get_network_type(self):
        #   return collections.namedtuple('DQN_network', ['q_values'])
        # def _network_template(self, state):
        #   return self.network(self.num_actions, self._get_network_type(), state)

        def _create_network(self, name):
          """Builds the convolutional network used to compute the agent's Q-values.

          Args:
            name: str, this name is passed to the tf.keras.Model and used to create
              variable scope under the hood by the tf.keras.Model.
          Returns:
            network: tf.keras.Model, the network instantiated by the Keras model.
          """
          # `self.network` is set to `atari_lib.NatureDQNNetwork`.
          network = self.network(self.num_actions, name=name)
          return network

        def _build_networks(self):
          # The following two lines are replaced.
          # self.online_convnet = tf.make_template('Online', self._network_template)
          # self.target_convnet = tf.make_template('Target', self._network_template)
          self.online_convnet = self._create_network(name='Online')
          self.target_convnet = self._create_network(name='Target')
        ```

    *   If your code overwrites `._network_template()`, `._get_network_type()`
        or `._build_networks()` make sure you update your code to fit with the
        new API. If your code overwrites `._build_networks()` you need to
        replace `tf.make_template('Online', self._network_template)` with
        `self._create_network(name='Online')`.

    *   The variables of each network can be obtained from the networks as
        follows: `vars = self.online_convnet.variables`.

    *   Baselines and older checkpoints can be loaded by adding the following
        line to your gin file.

        ```
        atari_lib.maybe_transform_variable_names.legacy_checkpoint_load = True
        ```

*   **11/06/2019:** Visualization utilities added to generate videos and still
    images of a trained agent interacting with its environment. See an example
    colaboratory
    [here](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/agent_visualizer.ipynb).

*   **30/01/2019:** Dopamine 2.0 now supports general discrete-domain gym
    environments.

*   **01/11/2018:** Download links for each individual checkpoint, to avoid
    having to download all of the checkpoints.

*   **29/10/2018:** Graph definitions now show up in Tensorboard.

*   **16/10/2018:** Fixed a subtle bug in the IQN implementation and upated the
    colab tools, the JSON files, and all the downloadable data.

*   **18/09/2018:** Added support for double-DQN style updates for the
    `ImplicitQuantileAgent`.

    *   Can be enabled via the `double_dqn` constructor parameter.

*   **18/09/2018:** Added support for reporting in-iteration losses directly
    from the agent to Tensorboard.

    *   Set the `run_experiment.create_agent.debug_mode = True` via the
        configuration file or using the `gin_bindings` flag to enable it.
    *   Control frequency of writes with the `summary_writing_frequency` agent
        constructor parameter (defaults to `500`).

*   **27/08/2018:** Dopamine launched!
