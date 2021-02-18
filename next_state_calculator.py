## NN for calculating next state

import pygame
import random
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from time import sleep

BLACK = (31, 29, 36)
WHITE = (255, 255, 255)
pygame.font.init()


class NextStateCalc:
    def __init__(self):
        self.score = 0
        self.bag = None
        self.gap = 30
        self.width = 10
        self.stat = Stats()
        self.SPO_weights = np.random.rand(10, 5)
        self.velocities = 2 * np.random.rand(10, 5) - 1
        self.x = np.random.rand(10, 5)
        self.current_weights = self.SPO_weights
        self.weight_index = 0
        self.weight_perf = []
        self.update_perf = []
        self.update_step = False
        self.game_count = 0
        self.score_rec = 0
        self.initiate_game()

    def initiate_game(self):
        self.win = pygame.display.set_mode((500, 690))
        pygame.init()
        pygame.time.set_timer(pygame.USEREVENT, 1000)
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        self.game = True
        self.move_count = 0
        while self.game:
            self.create_board()
            self.win.fill(BLACK)
            self.run = True
            self.score = 0
            self.game_count += 1
            while self.run:
                print("--------------------------------")
                print("update step: ", self.update_step)
                print("weight perf: ", self.weight_perf)
                print("update perf: ", self.update_perf)
                print("weight index: ", self.weight_index)
                print("--------------------------------")

                ## get action
                piece_type, action = self.get_random_input()

                ## apply action
                self.apply_action(piece_type, action)
                self.move_count += 1
                ## clear rows
                self.board, self.num_full_lines = self.clear_full_lines(self.board)

                ## check if game ends
                if np.any(self.board[0:3, :] != 0):
                    self.score -= 5
                    self.score_rec += self.score
                    self.run = False
                else:
                    ## count score
                    self.score += self.num_full_lines * 100 + 1
                    ## update board
                    self.display_array_score()

                ## QUIT button
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.game = False
                            self.run = False
                sleep(0.1)
            """ after 5 games, go to next weight.
                after getting weight performance for all 10 weights, initiate update step.
                update step involves testing new x and new positions.

                if self.update_step is True, then score rec and move count is used to check 
                if position should be updated 
            """
            if not self.update_step:
                if self.game_count % 5 == 0 and self.game_count != 0:
                    self.weight_performance()
                if len(self.weight_perf) == 10:
                    self.update_velocity_and_x()
            else:
                if self.game_count % 5 == 0:
                    self.weight_performance()
                if len(self.update_perf) == 10:
                    self.update_weights()

    def display_array_score(self):
        self.win.fill(BLACK)
        ## draw boundaries
        pygame.draw.line(self.win, WHITE, (0, 3 * 30), (10 * 30, 3 * 30))
        pygame.draw.line(self.win, WHITE, (10 * 30, 3 * 30), (10 * 30, 23 * 30))
        ## draw blocks
        rows = [x for x in self.board]
        for row_index, row in enumerate(rows):
            for col_index, col in enumerate(row):
                if col != 0 and row_index >= 3:
                    pygame.draw.rect(
                        self.win,
                        WHITE,
                        (
                            col_index * 30,
                            row_index * 30,
                            30,
                            30,
                        ),
                        0,
                    )
        # Game Count
        font = pygame.font.SysFont("ComicSans", 40)
        text = font.render("Game Count", 1, (255, 255, 255))
        count = font.render(str(self.game_count), 1, (255, 255, 255))
        self.win.blit(text, (self.width * self.gap + int(self.gap / 2), 1 * self.gap))
        self.win.blit(count, (self.width * self.gap + int(self.gap / 2), 2 * self.gap))

        # Write Score
        font = pygame.font.SysFont("ComicSans", 40)
        text = font.render("Score", 1, (255, 255, 255))
        score = font.render(str(self.score), 1, (255, 255, 255))
        self.win.blit(text, (self.width * self.gap + int(self.gap / 2), 3 * self.gap))
        self.win.blit(score, (self.width * self.gap + int(self.gap / 2), 4 * self.gap))

        # Display weights
        font = pygame.font.SysFont("ComicSans", 25)
        text = font.render("Weights", 1, (255, 255, 255))
        weight0 = font.render(
            str(self.SPO_weights[self.weight_index][0])[0:5], 1, (255, 255, 255)
        )
        weight1 = font.render(
            str(self.SPO_weights[self.weight_index][1])[0:5], 1, (255, 255, 255)
        )
        weight2 = font.render(
            str(self.SPO_weights[self.weight_index][2])[0:5], 1, (255, 255, 255)
        )
        weight3 = font.render(
            str(self.SPO_weights[self.weight_index][3])[0:5], 1, (255, 255, 255)
        )
        weight4 = font.render(
            str(self.SPO_weights[self.weight_index][4])[0:5], 1, (255, 255, 255)
        )
        text_eval = font.render("weight performance", 1, (255, 255, 255))
        max_eval = font.render(
            str(max(self.weight_perf, default=0)), 1, (255, 255, 255)
        )

        self.win.blit(text, (self.width * self.gap + int(self.gap / 2), 5.3 * self.gap))
        self.win.blit(
            weight0, (self.width * self.gap + int(self.gap / 2), 6 * self.gap)
        )
        self.win.blit(
            weight1, (self.width * self.gap + int(self.gap / 2), 6.5 * self.gap)
        )
        self.win.blit(
            weight2, (self.width * self.gap + int(self.gap / 2), 7 * self.gap)
        )
        self.win.blit(
            weight3, (self.width * self.gap + int(self.gap / 2), 7.5 * self.gap)
        )
        self.win.blit(
            weight4, (self.width * self.gap + int(self.gap / 2), 8 * self.gap)
        )
        self.win.blit(
            text_eval, (self.width * self.gap + int(self.gap / 2), 9 * self.gap)
        )
        self.win.blit(
            max_eval, (self.width * self.gap + int(self.gap / 2), 9.5 * self.gap)
        )

        ## display x
        text = font.render("x Weights", 1, (255, 255, 255))
        weight0 = font.render(
            str(self.x[self.weight_index][0])[0:5], 1, (255, 255, 255)
        )
        weight1 = font.render(
            str(self.x[self.weight_index][1])[0:5], 1, (255, 255, 255)
        )
        weight2 = font.render(
            str(self.x[self.weight_index][2])[0:5], 1, (255, 255, 255)
        )
        weight3 = font.render(
            str(self.x[self.weight_index][3])[0:5], 1, (255, 255, 255)
        )
        weight4 = font.render(
            str(self.x[self.weight_index][4])[0:5], 1, (255, 255, 255)
        )

        self.win.blit(text, (self.width * self.gap + int(self.gap / 2), 10 * self.gap))
        self.win.blit(
            weight0, (self.width * self.gap + int(self.gap / 2), 10.5 * self.gap)
        )
        self.win.blit(
            weight1, (self.width * self.gap + int(self.gap / 2), 11 * self.gap)
        )
        self.win.blit(
            weight2, (self.width * self.gap + int(self.gap / 2), 11.5 * self.gap)
        )
        self.win.blit(
            weight3, (self.width * self.gap + int(self.gap / 2), 12 * self.gap)
        )
        self.win.blit(
            weight4, (self.width * self.gap + int(self.gap / 2), 12.5 * self.gap)
        )
        text_eval = font.render("x weight performance", 1, (255, 255, 255))
        max_eval = font.render(
            str(max(self.update_perf, default=0)), 1, (255, 255, 255)
        )

        self.win.blit(
            text_eval, (self.width * self.gap + int(self.gap / 2), 13.5 * self.gap)
        )
        self.win.blit(
            max_eval, (self.width * self.gap + int(self.gap / 2), 14 * self.gap)
        )

        ## display stats
        holes = font.render("Holes: ", 1, (255, 255, 255))
        num_holes = font.render(
            str(self.stat.num_holes(self.board)), 1, (255, 255, 255)
        )
        self.win.blit(holes, (self.width * self.gap + int(self.gap / 2), 15 * self.gap))
        self.win.blit(num_holes, (self.width * self.gap + self.gap * 3, 15 * self.gap))
        pygame.display.update()

    def apply_action(self, piece_type, action):
        """ board array, piece array, rot and col to create new board """
        self.board, _ = self.place_board(self.board, piece_type, action)

    def place_board(self, board, piece_type, action):
        rot = action[0]
        col = action[1]

        ## create piece and push to left edge
        piece = self.piece_array(piece_type, rot)
        piece = self.trim_piece(piece)
        p_shape = piece.shape

        ## place piece in board when possible
        row = 0

        X = board[(row + 1) : (row + 1 + p_shape[0]), col : (col + p_shape[1])] + piece

        while 2 not in X:
            row += 1
            if row + p_shape[0] >= 23:
                break
            X = (
                board[(row + 1) : (row + 1 + p_shape[0]), col : (col + p_shape[1])]
                + piece
            )

        board[row : (row + p_shape[0]), col : (col + p_shape[1])] = (
            board[row : (row + p_shape[0]), col : (col + p_shape[1])] + piece
        )
        return board, row

    def piece_array(self, piece_type, rot):
        if piece_type == 1:
            piece = np.zeros((4, 4))
            piece[1, :] = 1
        elif piece_type == 4:
            piece = np.ones((2, 2))
        else:
            piece = np.zeros((3, 3))
            if piece_type == 2:
                piece[1, :] = 1
                piece[0, 0] = 1
            elif piece_type == 3:
                piece[1, :] = 1
                piece[0, 2] = 1
            elif piece_type == 5:
                piece[0, 1:] = 1
                piece[1, 0:2] = 1
            elif piece_type == 6:
                piece[1, :] = 1
                piece[0, 1] = 1
            elif piece_type == 7:
                piece[0, 0:2] = 1
                piece[1, 1:] = 1

        for rotate in range(rot):
            piece = np.rot90(piece)

        return piece

    def trim_piece(self, piece):
        piece = piece[np.any(piece == 1, axis=1), :]
        piece = piece[:, np.any(piece == 1, axis=0)]
        return piece

    def create_board(self):
        self.board = np.zeros((23, 10))

    def piece_bag(self):
        bag = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
        random.shuffle(bag)
        return bag

    def get_next_piece(self):
        if not self.bag:
            self.bag = self.piece_bag()
        piece = self.bag[-1]
        del self.bag[-1]
        return piece

    def get_random_input(self):
        piece_type = self.get_next_piece()
        (rot, col) = self.SPO_get_action(piece_type)
        # rot = random.randint(0, 3)

        # piece = self.piece_array(piece_type, rot)
        # col = random.randint(0, 10 - piece.shape[1])

        return piece_type, (rot, col)

    def clear_full_lines(self, board):
        full_lines = np.where(np.all(board == 1, axis=1))
        num_full_lines = 0

        if full_lines[0].shape[0] != 0:
            num_full_lines = full_lines[0].shape[0]
            board = np.delete(board, full_lines, axis=0)
            board = np.vstack((np.zeros((num_full_lines, board.shape[1])), board))

        return board, num_full_lines

    def SPO_get_action(self, piece_type):
        stat = self.stat
        action_score = []
        for rot in range(4):
            piece = self.piece_array(piece_type, rot)
            for col in range(0, 11 - piece.shape[1]):
                board = self.board.copy()
                board, row = self.place_board(board, piece_type, (rot, col))
                _, num_full_lines = self.clear_full_lines(board)

                action_score.append(
                    (
                        (rot, col),
                        self.evaluate(
                            np.array(
                                [
                                    stat.num_holes(board),
                                    stat.row_transition(board),
                                    stat.col_transition(board),
                                    stat.well_sums(board),
                                    stat.bumpiness(board),
                                ]
                            )
                        ),
                    )
                )

        best_config = max(action_score, key=lambda x: x[1])
        best_action = best_config[0]
        self.max_eval = best_config[1]
        return best_action

    def initiate_neural_network(self):
        self.neural = MLPRegressor((100, 50), max_iter=1000)

    def evaluate(self, stat):
        return -np.inner(self.current_weights[self.weight_index], stat)

    def weight_performance(self):
        """ with score_rec and move count to calculate weight perf"""
        if self.update_step:
            self.update_perf.append(self.score_rec / self.move_count)
        else:
            self.weight_perf.append(self.score_rec / self.move_count)

        self.score_rec = 0
        self.move_count = 0
        self.weight_index = (self.weight_index + 1) % 10

    def update_velocity_and_x(self):
        """ find best weights g, update positions and velocities """

        best_value = max(self.weight_perf)
        index_of_best = self.weight_perf.index(best_value)
        best_weight = self.SPO_weights[index_of_best, :]

        r1 = random.random()
        r2 = random.random()

        ## update velocity

        self.velocities = 0.9 * self.velocities + 0.1 * r1 * (
            np.array(
                [
                    best_weight,
                ]
                * 10
            )
            - self.x
        )

        ## update x
        self.x = self.x + 0.2 * self.velocities
        self.update_step = True
        self.weight_index = 0
        self.current_weights = self.x

    def update_weights(self):

        indices_for_update = [
            i
            for i in range(len(self.weight_perf))
            if self.update_perf[i] > self.weight_perf[i]
        ]

        self.SPO_weights[indices_for_update, :] = self.x[indices_for_update, :]
        self.current_weights = self.SPO_weights
        self.update_step = False
        self.weight_index = 0
        self.weight_perf = []
        self.update_perf = []


class Stats:
    def __init__(self):
        pass

    def num_holes(self, board):
        sum_cols = sum(board)
        first_nonzero_row = np.argmax(board, axis=0)

        holes_in_col = (board.shape[0] - first_nonzero_row) - sum_cols
        holes_in_col = holes_in_col % 20
        import pdb

        pdb.set_trace()
        return sum(holes_in_col)

    def row_transition(self, board):
        row_trans = 0
        for row in board:
            if np.any(np.where(row)):
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
        col_sums = np.sum(board, axis=0)
        for col in range(len(col_sums) - 1):
            if col_sums[col + 1] == 0 and col_sums[col] != 0:
                well_sum += col_sums[col]

        return well_sum

    def bumpiness(self, board):
        col_sums = np.sum(board, axis=0)
        return np.sum(col_sums[1:] - col_sums[0:-1])


nsc = NextStateCalc()