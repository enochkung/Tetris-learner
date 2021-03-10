# Tetris AI

### Objective

#### The project is to use Deep Q learning to play Tetris.

### Method

#### For every step in which a new piece comes in, I check all possible rotations and columns. Apply a certain rotation and column yields a certain number of lines cleared, and a number of holes, aggregate height, bumpiness, and highest row of the new resulting board. These features are inputted in the model and a score is given. Action with the highest score is chosen to be the action

After applying the action, take the number of lines cleared, the features of the new resulting board, and record them as input. Then calculate both the reward and the maximum Q value of the new resulting board, and take reward + discount \* (max Q value of new board) as the output.

For 15 games, the input and outputs of each move is recorded and after 15 games, random records are chosen to train the model. Newer inputs are added to the old and once number of inputs is over 1000, most of them will be removed to welcome newer inputs.

### Reward

The reward for [num of cleared lines, num of holes, agg. height, bumpiness, highest row] is given as (num of cleared lines)^2 - (highest row)/100 if the action does not cause the game to be lost and -10 if the action loses the game.

### Notes

- MLPRegressor can be replaced with any other regression model, just as long as the methods such as .fit and .predict are legitimate methods
- discount rate is defined to be 0.4 \* 10/(10+game count) so that it decreases slowly over time

### Modules needed:

####

1. numpy

2. sklearn

3. pygame
