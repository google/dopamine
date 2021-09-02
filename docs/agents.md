# DQN And Rainbow


In the spirit of these principles, this first version focuses on supporting the
state-of-the-art, single-GPU *Rainbow* agent ([Hessel et al., 2018][rainbow])
applied to Atari 2600 game-playing ([Bellemare et al., 2013][ale]).
Specifically, our Rainbow agent implements the three components identified as
most important by [Hessel et al.][rainbow]:

*   n-step Bellman updates (see e.g. [Mnih et al., 2016][a3c])
*   Prioritized experience replay ([Schaul et al., 2015][prioritized_replay])
*   Distributional reinforcement learning ([C51; Bellemare et al., 2017][c51])

For completeness, we also provide an implementation of DQN ([Mnih et al.,
2015][dqn]).
