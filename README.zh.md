# Dopamine
[快速入门](#getting-started) |
[文档][docs] |
[基线结果][baselines] |
[更新日志](https://google.github.io/dopamine/docs/changelist)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<div align="center">
  <img src="https://google.github.io/dopamine/images/dopamine_logo.png"><br><br>
</div>

Dopamine 是一个专为强化学习（Reinforcement Learning）算法快速原型开发而设计的研究框架。它旨在满足研究者对一个轻量级、易于理解的代码库的需求，让用户能够在其中自由地尝试各种大胆创新的想法（探索性研究）。

我们的设计原则是：

* _易于实验（Easy experimentation）_：让新用户能够轻松运行基准测试实验。
* _灵活开发（Flexible development）_：让新用户能够轻松验证自己的研究构想。
* _精简可靠（Compact and reliable）_：为精选的、经过实战检验的算法提供可靠实现。
* _可复现性（Reproducible）_：促进实验结果的可复现性。特别是，我们的实验设置遵循了 [Machado et al. (2018)][machado] 给出的规范建议。

Dopamine 支持以下基于 JAX 实现的智能体（Agent）：

* DQN ([Mnih et al., 2015][dqn])
* C51 ([Bellemare et al., 2017][c51])
* Rainbow ([Hessel et al., 2018][rainbow])
* IQN ([Dabney et al., 2018][iqn])
* SAC ([Haarnoja et al., 2018][sac])
* PPO ([Schulman et al., 2017][ppo])

有关可用智能体的更多信息，请参阅[官方文档](https://google.github.io/dopamine/docs)。

上述许多智能体同时还保留了 TensorFlow（旧版）实现，但新加入的智能体通常会仅基于 JAX 实现。

免责声明：这不是 Google 的官方产品。

<a id="getting-started"></a>
## 快速入门

我们提供了用于运行 Dopamine 的 Docker 容器。
相关说明可参阅[此处](https://google.github.io/dopamine/docker/)。

此外，也可以从源代码安装 Dopamine（推荐方式）或通过 pip 安装。无论采用哪种方式，请先继续阅读下方的先决条件。

### 先决条件

Dopamine 支持 Atari 环境和 MuJoCo 环境。请在安装 Dopamine 之前先安装您计划使用的环境：

**Atari**

1. 现在这些环境通常随 [ale_py](https://github.com/Farama-Foundation/Arcade-Learning-Environment) 一起打包提供。
1. 您可能需要手动执行一些步骤以正确安装 `baselines`，详见[安装说明](https://github.com/openai/baselines)。

**MuJoCo**

1. 在[此处](https://github.com/openai/mujoco-py#install-mujoco)安装 MuJoCo 并获取许可证。
2. 运行 `pip install mujoco-py`（我们建议使用[虚拟环境][virtualenv]）。

### 从源代码安装

使用 Dopamine 最常见的方式是从源代码进行安装，并直接修改源代码：

```
git clone https://github.com/google/dopamine
```

克隆完成后，安装依赖项：

```
pip install -r dopamine/requirements.txt
```

Dopamine 支持基于 TensorFlow（旧版）和 JAX（积极维护中）的智能体。
有关安装 TensorFlow 的更多信息，请参阅 [TensorFlow 文档](https://www.tensorflow.org/install)。

注意：在使用 Dopamine 时，我们建议使用[虚拟环境][virtualenv]。

### 使用 Pip 安装

注意：对于大多数用户，我们强烈建议从源代码进行安装。

通过 pip 安装十分简单，但 Dopamine 的设计初衷是直接修改源码以开展实验。如果您需要编写自己的实验，建议从源代码安装。

```
pip install dopamine-rl
```

### 运行测试

您可以在 Dopamine 根目录下运行以下命令，测试安装是否成功：

```
export PYTHONPATH=$PYTHONPATH:$PWD
python -m tests.dopamine.atari_init_test
```

## 下一步

查阅[文档][docs]以获取有关训练智能体的更多信息。

我们为每一个 Dopamine 智能体提供了[基线结果（Baselines）][baselines]。

我们还提供了一系列 [Colaboratory 笔记本](https://github.com/google/dopamine/tree/master/dopamine/colab)，用于演示如何使用 Dopamine。

## 参考文献

[Bellemare et al., *The Arcade Learning Environment: An evaluation platform for
general agents*. Journal of Artificial Intelligence Research, 2013.][ale]

[Machado et al., *Revisiting the Arcade Learning Environment: Evaluation
Protocols and Open Problems for General Agents*, Journal of Artificial
Intelligence Research, 2018.][machado]

[Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning*.
Proceedings of the AAAI Conference on Artificial Intelligence, 2018.][rainbow]

[Mnih et al., *Human-level Control through Deep Reinforcement Learning*. Nature,
2015.][dqn]

[Schaul et al., *Prioritized Experience Replay*. Proceedings of the International
Conference on Learning Representations, 2016.][prioritized_replay]

[Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*,
arXiv preprint arXiv:1812.05905, 2018.][sac]

[Schulman et al., *Proximal Policy Optimization Algorithms*.][ppo]

## 致谢与引用

如果您在研究工作中使用了 Dopamine，请引用我们的[白皮书][dopamine_paper]。以下为 BibTeX 引用示例：

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

[docs]: https://google.github.io/dopamine/docs/
[baselines]: https://google.github.io/dopamine/baselines
[machado]: https://jair.org/index.php/jair/article/view/11182
[ale]: https://jair.org/index.php/jair/article/view/10819
[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[a3c]: http://proceedings.mlr.press/v48/mniha16.html
[prioritized_replay]: https://arxiv.org/abs/1511.05952
[c51]: http://proceedings.mlr.press/v70/bellemare17a.html
[rainbow]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17204/16680
[iqn]: https://arxiv.org/abs/1806.06923
[sac]: https://arxiv.org/abs/1812.05905
[ppo]: https://arxiv.org/abs/1707.06347
[dopamine_paper]: https://arxiv.org/abs/1812.06110
[vitualenv]: https://docs.python.org/3/library/venv.html#creating-virtual-environments
[virtualenv]: https://docs.python.org/3/library/venv.html#creating-virtual-environments

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月19日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
