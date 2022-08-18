# Connect4 game agent trained by self playing using [proximal policy optimization](https://arxiv.org/abs/1707.06347)
This is an implementation of a connect4 game agent which is trained by self playing against itself while using proximal policy optimization to learn optimal policy.

The agent was tested against three rule based agent random, weak, strong. Random make a random move, weak make move which has the highest sum of consecutive same stones in all 8 directions, strong has 3 prioirties which are make winning move or make lose avoiding move or make best move same as weak.

## Self play
<img src="https://i.imgur.com/sKG4Lgo.png" width="500" >
<br/>
Result of self play showed the agent was able to learn some rules of the game as it perform good against random opponnet but against weak and strong opponent it was good enough. Reason could be policy archietecture, exploration holding back(policy entropy) or limitation of algorithm.

## Against rule based opponent
<img src="https://i.imgur.com/N76Zg36.png" width="500" >
<br/>
Although self playing agent did not performed well against weak or strong rule based opponent, It perform well against these opponents when trained directly against these opponents, showing agent is capable of finding vulnerablities in rule base opponents.

## Requirements:
The program is testing with following dependencies
- torch   1.11.0+cu113