use std::{sync::mpsc, thread};

use candle_core::{Device, Result};

use crate::{
    mcts::MCTS,
    model::{
        queue::evaluation_thread,
        vit::{ViT, ViTConfig},
    },
    train::data::{GameHistory, GameState},
};

pub struct AlphaZeroSelfPlay<const NUM_WORKERS: usize> {
    vit: ViT,
}

impl<const NUM_WORKERS: usize> AlphaZeroSelfPlay<NUM_WORKERS> {
    pub fn new(vit_config: &ViTConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            vit: ViT::from_config(vit_config, device)?,
        })
    }

    // TODO: Add a temperature scheduler
    pub fn generate_history(
        &self,
        sims_per_move: usize,
        temperature: f32,
        device: &Device,
    ) -> GameHistory {
        let states = Vec::new();

        let mcts = MCTS::<NUM_WORKERS>::new();

        let (queue_tx, queue_rx) = mpsc::channel();

        let queue_handle = thread::spawn(move || evaluation_thread(&self.vit, queue_rx, device));

        while mcts.board().winner().is_none() {
            mcts.run_simulations(sims_per_move, queue_tx.clone());

            states.push(GameState {
                board: mcts.board().clone(),
                distribution: mcts.get_distribution(),
            });

            let action = mcts.sample_action(temperature);
            mcts.make_action(&action);
        }

        drop(queue_tx);

        queue_handle.join();

        GameHistory {
            states,
            winner: mcts.board().winner.unwrap(),
        }
    }
}
