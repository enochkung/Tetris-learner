## board mechanics
import numpy as np


class Board:
    def place_board(board, piece_type, action):
        rot = action[0]
        col = action[1]

        ## create piece and push to left edge
        piece = Board.piece_array(piece_type, rot)
        piece = Board.trim_piece(piece)
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

    def piece_array(piece_type, rot):
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

    def trim_piece(piece):
        piece = piece[np.any(piece == 1, axis=1), :]
        piece = piece[:, np.any(piece == 1, axis=0)]
        return piece

    def clear_full_lines(board):
        full_lines = np.where(np.all(board == 1, axis=1))
        num_full_lines = 0

        if full_lines[0].shape[0] != 0:
            num_full_lines = full_lines[0].shape[0]
            board = np.delete(board, full_lines, axis=0)
            board = np.vstack((np.zeros((num_full_lines, board.shape[1])), board))

        return board, num_full_lines
