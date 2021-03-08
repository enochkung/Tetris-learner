## decision trees

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
from time import sleep

BLACK = (31, 29, 36)
WHITE = (177, 156, 217)  # (255, 255, 255)
pygame.font.init()

"""
One cycle:

1. Start with random inputs
2. Calculate all stats into a vector and record it, the current piece, and the action to input array
3. Record the score and put to output array
4. Once reaching 500 records, train model.
5. In new cycle, inputs are created with epsilon chance of random and 1 - epsilon chance of being the action that model generates the highest score
 """


class TreeTetris:
    def __init__(self):
        self.score = 0
        self.bag = None
        self.gap = 30
        self.width = 10
        self.stat = Stats()
        self.input_record = []
        self.output_record = []
        self.weight_index = 0
        self.weight_perf = []
        self.update_perf = []
        self.update_step = False
        self.game_count = 0
        self.score_rec = 0
        self.alpha = 0.9
        self.gamma = 0.9

        ## initiate tree: tree predicts Q value of input
        # self.tetris_tree = DecisionTreeRegressor(min_samples_leaf=10)
        self.tetris_tree = MLPRegressor(hidden_layer_sizes=(50, 20, 10))
        ## run game
        self.initiate_game()

    def initiate_game(self):
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
            self.game_count += 1
            self.lines_cleared = 0
            self.move_count = 0
            self.game_input = []
            self.game_output = []
            while self.run:
                ## get action: random or optained by tree
                piece_type, action = self.get_action()
                pre_action_board = self.board.copy()

                ## apply action
                self.apply_action(piece_type, action)
                self.move_count += 1

                ## clear rows
                self.board, self.num_full_lines = self.clear_full_lines(self.board)

                ## check if game ends
                if np.any(self.board[0:3, :] != 0):
                    self.score -= 10
                    """record input and output"""
                    self.update_records(
                        -10, pre_action_board, piece_type, action, dump=True
                    )

                    self.score_rec += self.score
                    self.run = False
                else:
                    ## count score
                    try:
                        self.score += (
                            self.num_full_lines * 100
                            - (23 - np.where(np.any(self.board, axis=1))[0][0])
                            + self.move_count * 2
                        )
                        self.update_records(
                            self.num_full_lines * 100
                            - (23 - np.where(np.any(self.board, axis=1))[0][0])
                            + self.move_count * 2,
                            pre_action_board,
                            piece_type,
                            action,
                        )
                    except:
                        self.score += self.num_full_lines * 100 + self.move_count * 2
                        """record input and output"""
                        self.update_records(
                            self.num_full_lines * 100 + self.move_count * 2,
                            pre_action_board,
                            piece_type,
                            action,
                        )

                    ## count cleared lines
                    self.lines_cleared += self.num_full_lines
                    ## update board
                    self.display_array_score()

                ## QUIT, PAUSE, LOAD PREVIOUS LEARNING button
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.game = False
                            self.run = False
                        elif event.key == pygame.K_p:
                            self.pause_screen()

    def get_action(self):

        ## get next piece
        piece_type = self.get_next_piece()

        ep = random.random()
        if self.game_count < 300:
            ## if beginning or 1% of cases generate a random action
            rot = self.move_count % 4
            piece = self.piece_array(piece_type, rot)
            col = (2 * self.move_count) % (10 - piece.shape[1] + 1)
        elif ep < max(0.6 / np.log(self.game_count), 0.05):
            ## if beginning or 1% of cases generate a random action
            # rot = random.randint(0, 3)
            # piece = self.piece_array(piece_type, rot)
            # col = random.randint(0, 10 - piece.shape[1])
            rot = self.move_count % 4
            piece = self.piece_array(piece_type, rot)
            col = (2 * self.move_count) % (10 - piece.shape[1] + 1)
        else:
            ## action generated by tree
            action_and_predict_score = self.get_tree_action(piece_type)
            rot = action_and_predict_score[0]
            col = action_and_predict_score[1]

        return piece_type, (rot, col)

    def get_tree_action(self, piece_type):
        """ choose action that generates the highest score given by tree when inputted with """
        predict_score_collection = []
        for rot in range(4):
            piece = self.piece_array(piece_type, rot)
            for col in range(10 - piece.shape[1] + 1):
                test_vector = list(self.board.flatten())
                test_vector.extend([piece_type, rot, col])

                ## apply action
                predict_score = self.tetris_tree.predict(normalize([test_vector]))[0]
                predict_score_collection.append((rot, col, predict_score))

        return max(predict_score_collection, key=lambda y: y[2])

    def training_tree(self):
        print("trained")
        self.tetris_tree.fit(self.input_record, self.output_record)

    def update_records(self, score, pre_action_board, piece_type, action, dump=False):
        if len(self.input_record) >= 5000:
            normalize(self.input_record)
            print([x[0] for x in self.output_record if x[0] < 0 or x[0] > 80])
            self.training_tree()
            ## keep some old records
            self.keep_old_records()

        ## add current board and score to input and output
        stat_vector = list(pre_action_board.flatten())
        stat_vector.extend([piece_type, action[0], action[1]])
        self.game_input.append(stat_vector)
        try:
            self.game_output.append(
                [
                    self.tetris_tree.predict(stat_vector) * (1 - self.alpha)
                    + self.alpha * (score + self.gamma * self.max_next_q(self.board))
                ]
            )
        except:
            self.game_output.append([score])

        if dump:
            ## record stats, piece, and action as input
            self.input_record.extend(self.game_input)
            ## record score as output
            self.output_record.extend(self.game_output)
            ## erase game input and output
            self.game_input = []
            self.game_output = []

    def keep_old_records(self):
        """ keep the 500 records with the largest scores """
        large_rec = [
            (self.input_record[i], self.output_record[i])
            for i in range(len(self.input_record))
            if self.output_record[i][0] > 80
        ]

        if large_rec:
            self.input_record = [x[0] for x in large_rec]
            self.output_record = [x[1] for x in large_rec]
            while len(self.input_record) < 100:
                self.input_record.extend([x[0] for x in large_rec])
                self.output_record.extend([x[1] for x in large_rec])
        else:
            self.input_record = []
            self.output_record = []

    def max_next_q(self, board):
        flat_board = list(board.flatten())
        predict_score_collection = []
        for piece_type in range(7):
            for rot in range(4):
                piece = self.piece_array(piece_type, rot)
                for col in range(10 - piece.shape[1] + 1):
                    test_input = flat_board.copy()
                    flat_board.extend([piece_type, rot, col])
                    predict_score_collection.append(
                        self.tetris_tree.predict(flat_board)[0]
                    )
        return max(predict_score_collection)

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
        # for row_index, row in enumerate(rows):
        #     for col_index, col in enumerate(row):
        #         if col != 0 and row_index >= 3:
        #             pygame.draw.rect(
        #                 self.win,
        #                 BLACK,
        #                 (
        #                     col_index * 30,
        #                     row_index * 30,
        #                     30,
        #                     30,
        #                 ),
        #                 2,
        #             )
        # Game Count
        font = pygame.font.SysFont("ComicSans", 40)
        text = font.render("Game Count", 1, (255, 255, 255))
        count = font.render(str(self.game_count), 1, (255, 255, 255))
        self.win.blit(text, (self.width * self.gap + int(self.gap / 2), 1 * self.gap))
        self.win.blit(count, (self.width * self.gap + int(self.gap / 2), 2 * self.gap))

        # Write Score
        font = pygame.font.SysFont("ComicSans", 40)
        text = font.render("Lines Cleared", 1, (255, 255, 255))
        score = font.render(str(self.lines_cleared), 1, (255, 255, 255))
        self.win.blit(text, (self.width * self.gap + int(self.gap / 2), 3 * self.gap))
        self.win.blit(score, (self.width * self.gap + int(self.gap / 2), 4 * self.gap))

        ## display stats
        font = pygame.font.SysFont("ComicSans", 20)
        holes = font.render("holes: ", 1, (255, 255, 255))
        num_holes = font.render(
            str(self.stat.num_holes(self.board)), 1, (255, 255, 255)
        )
        self.win.blit(holes, (self.width * self.gap + int(self.gap / 2), 16 * self.gap))
        self.win.blit(num_holes, (self.width * self.gap + self.gap * 4, 16 * self.gap))

        holes = font.render("row trans: ", 1, (255, 255, 255))
        num_holes = font.render(
            str(self.stat.row_transition(self.board)), 1, (255, 255, 255)
        )
        self.win.blit(
            holes, (self.width * self.gap + int(self.gap / 2), 16.5 * self.gap)
        )
        self.win.blit(
            num_holes, (self.width * self.gap + self.gap * 4, 16.5 * self.gap)
        )

        holes = font.render("col trans: ", 1, (255, 255, 255))
        num_holes = font.render(
            str(self.stat.col_transition(self.board)), 1, (255, 255, 255)
        )
        self.win.blit(holes, (self.width * self.gap + int(self.gap / 2), 17 * self.gap))
        self.win.blit(num_holes, (self.width * self.gap + self.gap * 4, 17 * self.gap))

        holes = font.render("well sums: ", 1, (255, 255, 255))
        num_holes = font.render(
            str(self.stat.well_sums(self.board)), 1, (255, 255, 255)
        )
        self.win.blit(
            holes, (self.width * self.gap + int(self.gap / 2), 17.5 * self.gap)
        )
        self.win.blit(
            num_holes, (self.width * self.gap + self.gap * 4, 17.5 * self.gap)
        )

        # holes = font.render("landing: ", 1, (255, 255, 255))
        # num_holes = font.render(
        #     str(self.stat.bumpiness(self.board)), 1, (255, 255, 255)
        # )
        # self.win.blit(holes, (self.width * self.gap + int(self.gap / 2), 18 * self.gap))
        # self.win.blit(num_holes, (self.width * self.gap + self.gap * 4, 18 * self.gap))
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

    def clear_full_lines(self, board):
        full_lines = np.where(np.all(board == 1, axis=1))
        num_full_lines = 0

        if full_lines[0].shape[0] != 0:
            num_full_lines = full_lines[0].shape[0]
            board = np.delete(board, full_lines, axis=0)
            board = np.vstack((np.zeros((num_full_lines, board.shape[1])), board))

        return board, num_full_lines

    def pause_screen(
        self,
    ):
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        run = False


class Stats:
    def __init__(self):
        pass

    def all_stat_in_vector(self, board):
        return [
            self.num_holes(board),
            self.row_transition(board),
            self.col_transition(board),
            self.well_sums(board),
            self.bumpiness(board),
        ]

    def num_holes(self, board):
        sum_cols = sum(board)
        first_nonzero_row = np.argmax(board, axis=0)

        holes_in_col = (board.shape[0] - first_nonzero_row) - sum_cols
        holes_in_col = holes_in_col % 23
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


nsc = TreeTetris()