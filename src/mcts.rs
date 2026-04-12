use std::{iter, mem};

use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rng,
};

use crate::{
    board::{Board, Player, Winner, action::Action},
    model::evaluate_board,
};

const C_PUCT: f32 = 4.0;
const VIRTUAL_LOSS: f32 = -3.0;

#[derive(Clone)]
struct MCTSNode {
    visit_count: Vec<usize>,
    prior: Vec<f32>,
    total_action_value: Vec<f32>,

    children: Vec<Option<MCTSNode>>,
}

impl MCTSNode {
    fn new(prior: Vec<f32>) -> Self {
        let n = prior.len();

        Self {
            visit_count: Vec::from_iter(iter::repeat_n(0, n)),
            prior,
            total_action_value: Vec::from_iter(iter::repeat_n(0., n)),

            children: Vec::from_iter(iter::repeat_n(None, n)),
        }
    }

    fn best_child_idx(&self) -> usize {
        let sqrt_total_visits = (self.visit_count.iter().sum::<usize>() as f32).sqrt();

        (0..self.prior.len())
            .map(|idx| {
                (if self.visit_count[idx] == 0 {
                    0.
                } else {
                    self.total_action_value[idx] / self.visit_count[idx] as f32
                }) + C_PUCT
                    * self.prior[idx]
                    * (sqrt_total_visits / (1. + self.visit_count[idx] as f32))
            })
            .enumerate()
            .max_by(|(_, v1), (_, v2)| {
                if v1 < v2 {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap()
            .0
    }

    fn run_simulation(&mut self, mut board: Board) -> f32 {
        if let Some(winner) = board.winner() {
            return match winner {
                Winner::Tie => 0.,
                Winner::Player(Player::Black) => 1.,
                Winner::Player(Player::White) => -1.,
            };
        }

        let player = board.player();

        let child_idx = self.best_child_idx();

        let action = board.legal_actions()[child_idx];

        board.make_action(&action);

        let action_value = if let Some(child) = self.children[child_idx].as_mut() {
            self.total_action_value[child_idx] += VIRTUAL_LOSS;

            child.run_simulation(board)
        } else {
            let (prior, action_value) = evaluate_board(&board);

            self.children[child_idx] = Some(MCTSNode::new(prior));

            action_value
        };

        self.total_action_value[child_idx] += match player {
            Player::Black => action_value,
            Player::White => -action_value,
        } - VIRTUAL_LOSS;
        self.visit_count[child_idx] += 1;

        action_value
    }
}

pub struct MCTS {
    board: Board,
    root: MCTSNode,
}

impl MCTS {
    pub fn new() -> Self {
        let board = Board::new();

        let (prior, _) = evaluate_board(&board);

        Self {
            board,
            root: MCTSNode::new(prior),
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn run_simulations(&mut self, num_simulations: usize) {
        // TODO: Make parallel
        for _ in 0..num_simulations {
            self.root.run_simulation(self.board.clone());
        }
    }

    fn make_action_index(&mut self, action_idx: usize) {
        let action = self.board.legal_actions()[action_idx];

        self.board.make_action(&action);

        if self.root.children[action_idx].is_some() {
            let new_root = self.root.children.remove(action_idx).unwrap();

            let _ = mem::replace(&mut self.root, new_root);
        } else {
            let (prior, _) = evaluate_board(&self.board);

            self.root = MCTSNode::new(prior);
        }
    }

    pub fn sample_action(&self, temperature: f32) -> Action {
        let weights = self
            .root
            .visit_count
            .iter()
            .map(|c| (*c as f32).powf(1. / temperature))
            .collect::<Vec<_>>();

        let dist = WeightedIndex::new(&weights).unwrap();

        self.board.legal_actions()[dist.sample(&mut rng())]
    }

    pub fn make_action(&mut self, Action(x, y): &Action) {
        let action_idx = self
            .board
            .legal_actions()
            .into_iter()
            .enumerate()
            .find(|(_, Action(a_x, a_y))| a_x == x && a_y == y)
            .unwrap()
            .0;

        self.make_action_index(action_idx);
    }
}
