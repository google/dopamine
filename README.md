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

This is not an official Google product.

## What's new
*  **16/10/2018:** Fixed a subtle bug in the IQN implementation and upated
   the colab tools, the JSON files, and all the downloadable data.
*  **18/09/2018:** Added support for double-DQN style updates for the
   `ImplicitQuantileAgent`.
   *  Can be enabled via the `double_dqn` constructor parameter.
*  **18/09/2018:** Added support for reporting in-iteration losses directly from
   the agent to Tensorboard.
   *  Include the flag `--debug_mode` in your command line to enable it.
   *  Control frequency of writes with the `summary_writing_frequency`
      agent constructor parameter (defaults to `500`).
*  **27/08/2018:** Dopamine launched!

## Instructions
### Install via source
Installing from source allows you to modify the agents and experiments as
you please, and is likely to be the pathway of choice for long-term use.
These instructions assume that you've already set up your favourite package
manager (e.g. `apt` on Ubuntu, `homebrew` on Mac OS X), and that a C++ compiler
is available from the command-line (almost certainly the case if your favourite
package manager works).

The instructions below assume that you will be running Dopamine in a *virtual
environment*. A virtual environment lets you control which dependencies are
installed for which program; however, this step is optional and you may choose
to ignore it.

Dopamine is a Tensorflow-based framework, and we recommend you also consult
the [Tensorflow documentation](https://www.tensorflow.org/install)
for additional details.

Finally, these instructions are for Python 2.7. While Dopamine is Python 3
compatible, there may be some additional steps needed during installation.

#### Ubuntu

First set up the virtual environment:

```
sudo apt-get update && sudo apt-get install virtualenv
virtualenv --python=python2.7 dopamine-env
source dopamine-env/bin/activate
```

This will create a directory called `dopamine-env` in which your virtual
environment lives. The last command activates the environment.

Then, install the dependencies to Dopamine. If you don't have access to a
GPU, then replace `tensorflow-gpu` with `tensorflow` in the line below
(see [Tensorflow instructions](https://www.tensorflow.org/install/install_linux)
for details).

```
sudo apt-get update && sudo apt-get install cmake zlib1g-dev
pip install absl-py atari-py gin-config gym opencv-python tensorflow-gpu
```

During installation, you may safely ignore the following error message:
*tensorflow 1.10.1 has requirement numpy<=1.14.5,>=1.13.3, but you'll have
numpy 1.15.1 which is incompatible*.

Finally, download the Dopamine source, e.g.

```
git clone https://github.com/google/dopamine.git
```

#### Mac OS X

First set up the virtual environment:

```
pip install virtualenv
virtualenv --python=python2.7 dopamine-env
source dopamine-env/bin/activate
```

This will create a directory called `dopamine-env` in which your virtual
environment lives. The last command activates the environment.

Then, install the dependencies to Dopamine:

```
brew install cmake zlib
pip install absl-py atari-py gin-config gym opencv-python tensorflow
```

During installation, you may safely ignore the following error message:
*tensorflow 1.10.1 has requirement numpy<=1.14.5,>=1.13.3, but you'll have
numpy 1.15.1 which is incompatible*.

Finally, download the Dopamine source, e.g.

```
git clone https://github.com/google/dopamine.git
```

#### Running tests

You can test whether the installation was successful by running the following:

```
cd dopamine
export PYTHONPATH=${PYTHONPATH}:.
python tests/atari_init_test.py
```

The entry point to the standard Atari 2600 experiment is
[`dopamine/atari/train.py`](https://github.com/google/dopamine/blob/master/dopamine/atari/train.py).
To run the basic DQN agent,

```
python -um dopamine.atari.train \
  --agent_name=dqn \
  --base_dir=/tmp/dopamine \
  --gin_files='dopamine/agents/dqn/configs/dqn.gin'
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


### Install as a library
An easy, alternative way to install Dopamine is as a Python library:

```
# Alternatively brew install, see Mac OS X instructions above.
sudo apt-get update && sudo apt-get install cmake
pip install dopamine-rl
pip install atari-py
```

Depending on your particular system configuration, you may also need to install
zlib (see "Install via source" above).

#### Running tests
From the root directory, tests can be run with a command such as:

```
python -um tests.agents.rainbow.rainbow_agent_test
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

If you use Dopamine in your work, we ask that you cite this repository as a
reference. The preferred format (authors in alphabetical order) is:

Marc G. Bellemare, Pablo Samuel Castro, Carles Gelada, Saurabh Kumar, Subhodeep Moitra.
Dopamine, https://github.com/google/dopamine, 2018.



[machado]: https://jair.org/index.php/jair/article/view/11182
[ale]: https://jair.org/index.php/jair/article/view/10819
[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[a3c]: http://proceedings.mlr.press/v48/mniha16.html
[prioritized_replay]: https://arxiv.org/abs/1511.05952
[c51]: http://proceedings.mlr.press/v70/bellemare17a.html
[rainbow]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17204/16680
[iqn]: https://arxiv.org/abs/1806.06923
