mod vit;

use std::iter;

use crate::board::Board;

/*
 Evaluates a board position, returning the prior probabilities and non-subjective value
*/
pub fn evaluate_board(board: &Board) -> (Vec<f32>, f32) {
    let n = board.legal_actions().len();

    (
        Vec::from_iter(iter::repeat_n(1. / (n as f32), n)),
        (board.score() as f32) / 64.,
    )
}
