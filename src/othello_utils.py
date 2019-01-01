import numpy as np

def convert_feature_to_board(fea, side):
    cur = fea[0,:,:]
    opp = fea[1,:,:]
    return cur*side+opp*(-side)

def convert_board_to_feature(board, side):
    fea = np.zeros((2,8,8))
    fea[0,:,:] = (board == side)
    fea[1,:,:] = (board == -side)
    fea = np.array(fea,dtype=int)
    return fea

def convert_mv_tuple_to_ind(mv):
    return mv[0]*8+mv[1]

def convert_mv_ind_to_tuple(mv):
    if mv == -1:
        return (-1,-1)
    return (mv//8,mv%8)

