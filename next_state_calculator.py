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
        self.win = pygame.display.set_mode((500, 690))
        pygame.init()
        pygame.time.set_timer(pygame.USEREVENT, 1000)
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        self.game = True
        while self.game:
            self.create_board()
            self.win.fill(BLACK)
            self.run = True
            self.score = 0
            while self.run:
                ## get action
                piece_type, action = self.get_random_input()

                ## apply action
                self.apply_action(piece_type, action)

                ## clear rows
                _, num_full_lines = self.clear_full_lines(self.board)

                ## check if game ends
                if np.any(self.board[0:3, :] != 0):
                    self.run = False
                else:
                    ## count score
                    self.score += num_full_lines
                    ## update board
                    self.display_array()
                    sleep(0.4)

                ## QUIT button
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.game = False
                            self.run = False

    def display_array(self):
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
                    # pygame.draw.rect(
                    #     self.win,
                    #     BLACK,
                    #     (
                    #         col_index * 30,
                    #         row_index * 30,
                    #         30,
                    #         30,
                    #     ),
                    #     1,
                    # )
        pygame.display.update()

    def apply_action(self, piece_type, action):
        """ board array, piece array, rot and col to create new board """
        self.board = self.place_board(self.board, piece_type, action)

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

        while 2 not in X and row + 1 + p_shape[0] < 23:
            row += 1
            X = (
                board[(row + 1) : (row + 1 + p_shape[0]), col : (col + p_shape[1])]
                + piece
            )

        board[row : (row + p_shape[0]), col : (col + p_shape[1])] = (
            board[row : (row + p_shape[0]), col : (col + p_shape[1])] + piece
        )
        return board

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

    def get_random_input(self):
        piece_type = random.randint(1, 7)
        rot = random.randint(0, 3)

        piece = self.piece_array(piece_type, rot)
        col = random.randint(0, 10 - piece.shape[1])

        return piece_type, (rot, col)

    def clear_full_lines(self, board):
        full_lines = np.where(np.all(board == 1))
        num_full_lines = 0
        if bool(len(full_lines[0])):
            num_full_lines = full_lines[0].shape[0]

            board = np.delete(board, full_lines, axis=0)
            board = np.vstack((np.zeros((num_full_lines, board.shape[1])), board))

        return board, num_full_lines


nsc = NextStateCalc()