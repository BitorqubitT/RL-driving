<h1>🏎️ DQN & PPO for Custom Racing Game Environment</h1>

<p>This repository contains my implementation of <strong>Deep Q-Network (DQN)</strong> and <strong>Proximal Policy Optimization (PPO)</strong> for a custom racing game environment.</p>

<h2>📁 Notable Files & Folders</h2>

<p>Some files that may not be self-explanatory:</p>

<ul>
  <li><code>utils/create_checkpoints.py</code><br>
    - Script to draw checkpoints for a new track.<br>
    - Saves checkpoints to a file in the <code>checkpoints/</code> folder.
  </li>
  <li><code>utils/walls.py</code><br>
    - Calculates the wall positions of a track from an image.<br>
    - Saves wall data as a <code>.csv</code> file in the <code>track_info/</code> directory.
  </li>
  <li><code>main_self_play.py</code><br>
    - Use this script for debugging and manual driving on the track.
  </li>
</ul>

<h2>🎥 Demo</h2>

<p align="center">
  <a href="https://www.youtube.com/watch?v=MHJ9NWQA5M8">
    <img src="https://img.youtube.com/vi/MHJ9NWQA5M8/hqdefault.jpg" alt="Watch the video" />
  </a>
</p>


<p><img src="https://github.com/user-attachments/assets/16513033-3a17-43fb-97b8-074affe5a7d3" alt="demo gif"></p>

<p><img width="801" alt="results" src="https://github.com/user-attachments/assets/f30dda29-996c-4cd8-93ca-3ff36bd91d8f" /></p>

<h2>📚 References</h2>

<ul>
  <li><a href="https://arxiv.org/abs/1312.5602">DQN Paper (2013)</a></li>
  <li><a href="https://arxiv.org/abs/1707.06347">PPO Paper (2017)</a></li>
  <li><a href="https://aim-studios.itch.io/top-down-pixel-art-race-cars">Car Sprites (AIM Studios)</a></li>
  <li><a href="https://github.com/vwxyzjn/cleanrl">CleanRL GitHub</a></li>
  <li><a href="https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html">PyTorch PPO Tutorial</a></li>
  <li><a href="https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html">PyTorch Q-Learning Tutorial</a></li>
</ul>
