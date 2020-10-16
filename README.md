# Dopamine

<div align="center">
  <img src="https://google.github.io/dopamine/images/dopamine_logo.png"><br><br>
</div>

Dopamine is a research framework for fast prototyping of reinforcement learning
algorithms. It aims to fill the need for a small, easily grokked codebase in
which users can freely experiment with wild ideas (speculative research).

Our design principles are:

* _Easy experimentation_: Make it easy for new users to run benchmark
                          experiments.
* _Flexible development_: Make it easy for new users to try out research ideas.
* _Compact and reliable_: Provide implementations for a few, battle-tested
                          algorithms.
* _Reproducible_: Facilitate reproducibility in results. In particular, our
                  setup follows the recommendations given by
                  [Machado et al. (2018)][machado].

In the spirit of these principles, this first version focuses on supporting the
state-of-the-art, single-GPU *Rainbow* agent ([Hessel et al., 2018][rainbow])
applied to Atari 2600 game-playing ([Bellemare et al., 2013][ale]).
Specifically, our Rainbow agent implements the three components identified as
most important by [Hessel et al.][rainbow]:

* n-step Bellman updates (see e.g. [Mnih et al., 2016][a3c])
* Prioritized experience replay ([Schaul et al., 2015][prioritized_replay])
* Distributional reinforcement learning ([C51; Bellemare et al., 2017][c51])

For completeness, we also provide an implementation of DQN
([Mnih et al., 2015][dqn]).
For additional details, please see our
[documentation](https://github.com/google/dopamine/tree/master/docs).

We provide a set of [Colaboratory
notebooks](https://github.com/google/dopamine/tree/master/dopamine/colab)
which demonstrate how to use Dopamine.

We provide a [website](https://google.github.io/dopamine/baselines/plots.html)
which displays the learning curves for all the provided agents, on all the
games.

This is not an official Google product.

## What's new
* **16/102020:** Learning curves for the [QR-DQN JAX
  agent](https://github.com/google/dopamine/blob/master/dopamine/jax/agents/quantile/quantile_agent.py)
  have been added to the [baseline
  plots](https://google.github.io/dopamine/baselines/plots.html)!

* **03/08/2020:** Dopamine now supports [JAX](https://github.com/google/jax)
  agents! This includes an implementation of the Quantile Regression agent (QR-DQN)
  which has been a common request. Find out more in our
  [jax](https://github.com/google/dopamine/tree/master/dopamine/jax)
  subdirectory, which includes trained agent checkpoints.

* **27/07/2020:** Dopamine now runs on TensorFlow 2. However, Dopamine is still
  written as TensorFlow 1.X code. This means your project may need to explicity
  disable TensorFlow 2 behaviours with:

  ```
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  ```
  if you are using custom entry-point for training your agent. The migration to
  TensorFlow 2 also means that Dopamine no longer supports Python 2.

*  **02/09/2019:** Dopamine has switched its network definitions to use
  tf.keras.Model. The previous `tf.contrib.slim` based networks are removed.
  If your agents inherit from dopamine agents you need to update your code.
   * `._get_network_type()` and `._network_template()` functions are replaced
      with `._create_network()` and `network_type` definitions are moved inside
      the model definition.

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
   * If your code overwrites `._network_template()`, `._get_network_type()` or
     `._build_networks()` make sure you update your code to fit with the new
     API. If your code overwrites `._build_networks()` you need to replace
     `tf.make_template('Online', self._network_template)` with
     `self._create_network(name='Online')`.
   * The variables of each network can be obtained from the networks as follows:
     `vars = self.online_convnet.variables`.
   * Baselines and older checkpoints can be loaded by adding the following line
     to your gin file.

     ```
     atari_lib.maybe_transform_variable_names.legacy_checkpoint_load = True
     ```
*  **11/06/2019:** Visualization utilities added to generate videos and still
   images of a trained agent interacting with its environment. See an example
   colaboratory
   [here](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/agent_visualizer.ipynb).
*  **30/01/2019:** Dopamine 2.0 now supports general discrete-domain gym
   environments.
*  **01/11/2018:** Download links for each individual checkpoint, to avoid
   having to download all of the checkpoints.
*  **29/10/2018:** Graph definitions now show up in Tensorboard.
*  **16/10/2018:** Fixed a subtle bug in the IQN implementation and upated
   the colab tools, the JSON files, and all the downloadable data.
*  **18/09/2018:** Added support for double-DQN style updates for the
   `ImplicitQuantileAgent`.
   *  Can be enabled via the `double_dqn` constructor parameter.
*  **18/09/2018:** Added support for reporting in-iteration losses directly from
   the agent to Tensorboard.
   *  Set the `run_experiment.create_agent.debug_mode = True` via the
      configuration file or using the `gin_bindings` flag to enable it.
   *  Control frequency of writes with the `summary_writing_frequency`
      agent constructor parameter (defaults to `500`).
*  **27/08/2018:** Dopamine launched!

## Instructions
### Install via source
Installing from source allows you to modify the agents and experiments as
you please, and is likely to be the pathway of choice for long-term use. The
instructions below assume that you will be running Dopamine in a *virtual
environment*. A virtual environment lets you control which dependencies are
installed for which program.

Dopamine is a Tensorflow-based framework, and we recommend you also consult
the [Tensorflow documentation](https://www.tensorflow.org/install)
for additional details. Finally, these instructions are for Python 3.6 and
above.

First download the Dopamine source.

```
git clone https://github.com/google/dopamine.git
```

Then create a virtual environment and activate it.

```
python3 -m venv ./dopamine-venv
source dopamine-venv/bin/activate
```

Finally setup the environment and install Dopamine's dependencies

```
pip install -U pip
pip install -r dopamine/requirements.txt
```

### Running tests

You can test whether the installation was successful by running the following:

```
cd dopamine
export PYTHONPATH=$PYTHONPATH:$PWD
python -m tests.dopamine.atari_init_test
```

### Training agents

#### Atari games

The entry point to the standard Atari 2600 experiment is
[`dopamine/discrete_domains/train.py`](https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/train.py).
To run the basic DQN agent,

```
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/dqn/configs/dqn.gin
```

By default, this will kick off an experiment lasting 200 million frames.
The command-line interface will output statistics about the latest training
episode:

```
[...]
I0824 17:13:33.078342 140196395337472 tf_logging.py:115] gamma: 0.990000
I0824 17:13:33.795608 140196395337472 tf_logging.py:115] Beginning training...
Steps executed: 5903 Episode length: 1203 Return: -19.
```

To get finer-grained information about the process,
you can adjust the experiment parameters in
[`dopamine/agents/dqn/configs/dqn.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin),
in particular by reducing `Runner.training_steps` and `Runner.evaluation_steps`,
which together determine the total number of steps needed to complete an
iteration. This is useful if you want to inspect log files or checkpoints, which
are generated at the end of each iteration.

More generally, the whole of Dopamine is easily configured using the
[gin configuration framework](https://github.com/google/gin-config).

#### Non-Atari discrete environments

We provide sample configuration files for training an agent on Cartpole and
Acrobot. For example, to train C51 on Cartpole with default settings, run the
following command:

```
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/rainbow/configs/c51_cartpole.gin
```

You can train Rainbow on Acrobot with the following command:

```
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/rainbow/configs/rainbow_acrobot.gin
```


### Install as a library
An easy, alternative way to install Dopamine is as a Python library:

```
pip install dopamine-rl
```

### References

[Bellemare et al., *The Arcade Learning Environment: An evaluation platform for
general agents*. Journal of Artificial Intelligence Research, 2013.][ale]

[Machado et al., *Revisiting the Arcade Learning Environment: Evaluation
Protocols and Open Problems for General Agents*, Journal of Artificial
Intelligence Research, 2018.][machado]

[Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning*.
Proceedings of the AAAI Conference on Artificial Intelligence, 2018.][rainbow]

[Mnih et al., *Human-level Control through Deep Reinforcement Learning*. Nature,
2015.][dqn]

[Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*. Proceedings
of the International Conference on Machine Learning, 2016.][a3c]

[Schaul et al., *Prioritized Experience Replay*. Proceedings of the International
Conference on Learning Representations, 2016.][prioritized_replay]

### Giving credit

If you use Dopamine in your work, we ask that you cite our
[white paper][dopamine_paper]. Here is an example BibTeX entry:

```
@article{castro18dopamine,
  author    = {Pablo Samuel Castro and
               Subhodeep Moitra and
               Carles Gelada and
               Saurabh Kumar and
               Marc G. Bellemare},
  title     = {Dopamine: {A} {R}esearch {F}ramework for {D}eep {R}einforcement {L}earning},
  year      = {2018},
  url       = {http://arxiv.org/abs/1812.06110},
  archivePrefix = {arXiv}
}
```



[machado]: https://jair.org/index.php/jair/article/view/11182
[ale]: https://jair.org/index.php/jair/article/view/10819
[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[a3c]: http://proceedings.mlr.press/v48/mniha16.html
[prioritized_replay]: https://arxiv.org/abs/1511.05952
[c51]: http://proceedings.mlr.press/v70/bellemare17a.html
[rainbow]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17204/16680
[iqn]: https://arxiv.org/abs/1806.06923
[dopamine_paper]: https://arxiv.org/abs/1812.06110
