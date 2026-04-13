use std::{
    iter, mem,
    sync::{Arc, Mutex, atomic::AtomicUsize, mpsc},
    thread,
};

use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rng,
};

use crate::{
    board::{Board, Player, Winner, action::Action},
    model::{evaluate_board, queue::EvaluateRequest},
};

const C_PUCT: f32 = 4.0;
const VIRTUAL_LOSS: i32 = 3;

struct MCTSNode {
    visit_count: Vec<isize>,
    prior: Vec<f32>,
    total_action_value: Vec<f32>,

    children: Vec<Option<Arc<Mutex<MCTSNode>>>>,
}

impl MCTSNode {
    fn new(prior: Vec<f32>) -> Self {
        let n = prior.len();

        Self {
            visit_count: Vec::from_iter(iter::repeat_n(0, n)),
            prior,
            total_action_value: Vec::from_iter(iter::repeat_n(0., n)),

            children: Vec::from_iter((0..n).map(|_| None)),
        }
    }

    fn best_child_idx(&self) -> usize {
        let sqrt_total_visits = (self.visit_count.iter().sum::<isize>() as f32).sqrt();

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

    fn run_simulation(node: Arc<Mutex<Self>>, mut board: Board) -> f32 {
        if let Some(winner) = board.winner() {
            return match winner {
                Winner::Tie => 0.,
                Winner::Player(Player::Black) => 1.,
                Winner::Player(Player::White) => -1.,
            };
        }

        let player = board.player();

        let child_idx = node.lock().unwrap().best_child_idx();

        let action = board.legal_actions()[child_idx];

        board.make_action(&action);

        {
            let mut node_guard = node.lock().unwrap();

            node_guard.total_action_value[child_idx] -= VIRTUAL_LOSS as f32;
            node_guard.visit_count[child_idx] += VIRTUAL_LOSS as isize;
        }

        let action_value = if let Some(child) = node.lock().unwrap().children[child_idx].clone() {
            Self::run_simulation(child, board)
        } else {
            let (prior, action_value) = evaluate_board(&board);

            node.lock().unwrap().children[child_idx] =
                Some(Arc::new(Mutex::new(MCTSNode::new(prior))));

            action_value
        };

        {
            let mut node_guard = node.lock().unwrap();

            node_guard.total_action_value[child_idx] += match player {
                Player::Black => action_value,
                Player::White => -action_value,
            } + VIRTUAL_LOSS as f32;
            node_guard.visit_count[child_idx] += 1 - VIRTUAL_LOSS as isize;
        };

        action_value
    }
}

pub struct MCTS<const NUM_WORKERS: usize> {
    board: Board,
    root: Arc<Mutex<MCTSNode>>,
}

impl<const NUM_WORKERS: usize> MCTS<NUM_WORKERS> {
    pub fn new() -> Self {
        let board = Board::new();

        let (prior, _) = evaluate_board(&board);

        Self {
            board,
            root: Arc::new(Mutex::new(MCTSNode::new(prior))),
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn run_simulations(
        &mut self,
        num_simulations: usize,
        queue_tx: mpsc::Sender<EvaluateRequest>,
    ) {
        let sim_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(NUM_WORKERS);

        for _ in 0..NUM_WORKERS {
            let worker_sim_count = sim_count.clone();
            let worker_root = self.root.clone();
            let worker_board = self.board.clone();

            handles.push(thread::spawn(move || {
                loop {
                    MCTSNode::run_simulation(worker_root.clone(), worker_board.clone());

                    if worker_sim_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        > num_simulations
                    {
                        break;
                    }
                }
            }));
        }

        for h in handles {
            let _ = h.join();
        }
    }

    pub fn get_distribution(&self) -> Vec<f32> {
        self.root
            .lock()
            .unwrap()
            .visit_count
            .iter()
            .map(|c| (*c as f32).powf(1. / temperature))
            .collect::<Vec<_>>()
    }

    fn make_action_index(&mut self, action_idx: usize) {
        let action = self.board.legal_actions()[action_idx];

        self.board.make_action(&action);

        if self.root.lock().unwrap().children[action_idx].is_some() {
            let new_root = self
                .root
                .lock()
                .unwrap()
                .children
                .remove(action_idx)
                .unwrap();

            let _ = mem::replace(&mut self.root, new_root);
        } else {
            let (prior, _) = evaluate_board(&self.board).unwrap();

            self.root = Arc::new(Mutex::new(MCTSNode::new(prior)));
        }
    }

    pub fn sample_action(&self, temperature: f32) -> Action {
        let dist = WeightedIndex::new(&self.get_distribution()).unwrap();

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
