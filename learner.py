# learner class

import pygame
import random
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from time import sleep
from board_mechanics import Board

""" the Learner class (sklearn MLPRegressor but can be changed accordingly) and Stat class that calculates stats of the board (number of holes, height, etc. )"""


class Learner(Board):
    """
    methods
    -------

    get_model_action: (board, piece_type)
    - get action with highest score

    apply_action_to_board: (board, piece, (rot,col))
    - outputs resulting board, after clearing rows, and output number of rows cleared

    get_reward: (cleared_lines)
    - outputs cleared_lines ^2

    board_to_model_input: (board, piece_type, (rot,col), cleared_lines)
    - after applying action to board, the stats of the resulting board and the number of cleared lines
    - [cleared lines, number of holes, bumpiness]

    input_format: (board, cleared_lines)
    - board is post action
    - apply get stats and combine with cleared_lines

    get_best_action: (action_score_collect) get a random action corresponding to a highest score

    update_record: (post_action_board, cleared_lines)
    - for inputs corresponding to board and cleared_lines, its output is the reward + average of
      Q values on application of all pieces and actions on post_action_board

    Q-value calculator: (board, piece, action)
    - apply action to board to get cleared lines and next board
    - get stat from next board
    - get average Q-value calculator on all possible action
    """

    def __init__(self):
        self.learner = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            max_iter=1000,
            learning_rate="adaptive",
            learning_rate_init=0.01,
            shuffle=True,
        )
        # self.learner = RandomForestRegressor()
        self.stat = Stats()
        self.discount = 0.4

    def get_best_action(self, preaction_board, piece_type):
        """ given board, find best action (rot,col) """

        action_score_pairs = []
        for rot in range(4):
            for col in range(self.col_range(piece_type, rot)):
                test_board = preaction_board.copy()
                score = self.board_action_to_predict_score(
                    test_board, piece_type, (rot, col)
                )
                if action_score_pairs:
                    if score == action_score_pairs[0][1]:
                        action_score_pairs.append(((rot, col), score))
                    elif score > action_score_pairs[0][1]:
                        action_score_pairs = [((rot, col), score)]
                else:
                    action_score_pairs = [((rot, col), score)]

        random_choose = random.randint(0, len(action_score_pairs) - 1)
        return action_score_pairs[random_choose]

    def update_record(self, postaction_board, cleared_lines, game_count):
        """ get reward and average q value for next state """
        ## put board in input form
        inputs = self.board_to_input(postaction_board, cleared_lines)

        ## get reward
        reward = self.get_reward(postaction_board, cleared_lines)
        if reward != -10:
            ## get average predicted score for next state
            score_collect = []
            for piece_type in range(7):
                for rot in range(4):
                    for col in range(self.col_range(piece_type, rot)):
                        score = self.board_action_to_predict_score(
                            postaction_board, piece_type, (rot, col)
                        )
                        score_collect.append(score)
            ## take max of scores
            max_score = max(score_collect)
        else:
            max_score = 0

        ## output
        output = reward + self.discount * 10 / (10 + game_count) * max_score

        return inputs, output

    def board_action_to_predict_score(self, board, piece_type, action):

        new_board = self.apply_action_to_board(board, piece_type, action)
        new_board, cleared_lines = self.clear_lines(new_board)
        try:
            inputs = self.board_to_input(new_board, cleared_lines)

            return self.predict_output(inputs)
        except:
            return self.get_reward(new_board, cleared_lines)

    def col_range(self, piece_type, rot):
        if piece_type == 1:
            if rot % 2 == 0:
                return 7
            else:
                return 10
        elif piece_type == 4:
            return 9
        else:
            if rot % 2 == 0:
                return 8
            else:
                return 9

        pass

    def apply_action_to_board(self, board, piece_type, action):
        """ output post action board """
        new_board, _ = Board.place_board(board, piece_type, action)
        return new_board

    def clear_lines(self, board):
        """ count and remove full lines """
        return Board.clear_full_lines(board)

    def board_to_input(self, board, cleared_lines):
        """ combine cleared_lines and number of holes, bumpiness, agg height """
        inputs = [cleared_lines]
        inputs.extend(self.stat.all_stat_in_vector(board))
        return inputs

    def predict_output(self, inputs):
        """ get model predicted score """
        return self.learner.predict(inputs)[0]

    def updating_model(self, inputs, outputs):
        """ fitting the model based on inputs and outputs """
        ## randomly select from experience and train
        train_inputs, _, train_outputs, _ = train_test_split(
            inputs, outputs, shuffle=True, train_size=0.7, random_state=10
        )
        self.learner.fit(normalize(train_inputs), train_outputs)

    def get_reward(self, board, cleared_lines):
        """ if board is a lost board, then return -10, else return 1 point for not losing and cleared_lines **2 * 10"""
        if self.board_lost(board):
            return -10
        return cleared_lines ** 2 * 10 - self.stat.highest_row(board) / 100

    def board_lost(self, board):
        """ board is lost if there are elements in the first three rows """
        return np.any(board[0:3, :] != 0)

    def print_model_pred(self, inputs, outputs):
        pred_outputs = self.learner.predict(normalize(inputs))

        print("mean square error:")
        print(np.sum((np.array(outputs) - np.array(pred_outputs)) ** 2))


class Stats:
    def __init__(self):
        pass

    def all_stat_in_vector(self, board):
        return [
            # self.num_completed_lines(board),
            self.num_holes(board),
            # self.row_transition(board),
            self.agg_height(board),
            # self.col_transition(board),
            # self.well_sums(board),
            self.bumpiness(board),
            self.highest_row(board),
        ]

    def num_completed_lines(self, board):
        return sum(np.all(board == 1, axis=1))

    def num_holes(self, board):
        sum_cols = sum(board)
        first_nonzero_row = np.argmax(board, axis=0)

        holes_in_col = (board.shape[0] - first_nonzero_row) - sum_cols
        holes_in_col = holes_in_col % 23
        # print("holes: ", sum(holes_in_col))
        return sum(holes_in_col)

    def row_transition(self, board):
        row_trans = 0
        for row in board:
            if np.any(np.where(row == 0)) and ~np.all(row == 0):
                for col_index in range(len(row)):
                    try:
                        if row[col_index] - row[col_index + 1] != 0:
                            row_trans += 1
                    except:
                        continue
        return row_trans

    def col_transition(self, board):
        return self.row_transition(board.T)

    def well_sums(self, board):
        well_sum = 0
        height = (23 - np.argmax(board, axis=0)) % 23

        if height[0] < height[1]:
            well_sum += height[1] - height[0]
        if height[-2] > height[-1]:
            well_sum += height[-2] - height[-1]

        well_sum += sum(
            [
                min(height[i - 1] - height[i], height[i + 1] - height[i])
                for i in range(1, 9)
                if height[i] < height[i + 1] and height[i] < height[i - 1]
            ]
        )

        return well_sum

    def bumpiness(self, board):
        turned_board = board.T
        col_sums = [(23 - np.argmax(col)) % 23 for col in turned_board]
        height_diff = [
            np.abs(col_sums[i + 1] - col_sums[i]) for i in range(len(col_sums) - 1)
        ]
        return sum(height_diff)

    def agg_height(self, board):
        return sum(
            [
                23 - np.where(row)[0][0]
                for row in board.T
                if 0 not in np.where(row)[0].shape
            ]
        )

    def highest_row(self, board):
        if len(np.where(np.any(board, axis=1))[0]) != 0:
            return 23 - np.where(np.any(board, axis=1))[0][0]
        return 0
