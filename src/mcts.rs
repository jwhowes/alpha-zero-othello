use std::iter;

use crate::{board::{Board, Player, Winner}, model::evaluate_board};

const C_PUCT: f32 = 4.0;
const VIRTUAL_LOSS: f32 = -3.0;

#[derive(Clone)]
struct MCTSNode {
    visit_count: Vec<usize>,
    prior: Vec<f32>,
    total_action_value: Vec<f32>,

    children: Vec<Option<MCTSNode>>
}

impl MCTSNode {
    fn new(prior: Vec<f32>) -> Self {
        let n = prior.len();

        Self {
            visit_count: Vec::from_iter(iter::repeat_n(0, n)),
            prior,
            total_action_value: Vec::from_iter(iter::repeat_n(0., n)),

            children: Vec::from_iter(iter::repeat_n(None, n))
        }
    }

    fn best_child_idx(&self, parent_visits: usize) -> usize {
        let sqrt_parent_visits = (parent_visits as f32).sqrt();

        (0..self.prior.len())
            .map(
                |idx| {
                    (
                        if self.visit_count[idx] == 0 {
                            0.
                        } else {
                            self.total_action_value[idx] / self.visit_count[idx] as f32
                        }
                    ) +
                    C_PUCT * self.prior[idx] * (
                        sqrt_parent_visits / (1. + self.visit_count[idx] as f32)
                    )
                }
            )
            .enumerate()
            .max_by(|(_, v1), (_, v2)| if v1 < v2 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            })
            .unwrap()
            .0
    }

    async fn run_simulation(&mut self, parent_visits: usize, mut board: Board) -> f32 {
        if let Some(winner) = board.winner() {
            return match winner {
                Winner::Tie => 0.,
                Winner::Player(Player::PlayerOne) => 1.,
                Winner::Player(Player::PlayerTwo) => -1.
            };
        }

        let player = board.player();

        let child_idx = self.best_child_idx(parent_visits);
        
        let action = board.legal_actions()[child_idx];

        board.make_action(&action);

        let action_value = if let Some(child) = self.children[child_idx].as_mut() {
            self.total_action_value[child_idx] += VIRTUAL_LOSS;

            Box::pin(child.run_simulation(self.visit_count[child_idx], board)).await
        } else {
            let (prior, action_value) = evaluate_board(&board).await;

            self.children[child_idx] = Some(MCTSNode::new(prior));

            action_value
        };

        self.total_action_value[child_idx] += match player {
            Player::PlayerOne => action_value,
            Player::PlayerTwo => -action_value
        } - VIRTUAL_LOSS;
        self.visit_count[child_idx] += 1;

        action_value
    }
}