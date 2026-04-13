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
    pub fn new(vit_config: ViTConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            vit: ViT::from_config(vit_config, device)?,
        })
    }

    pub fn generate_history(
        &self,
        sims_per_move: usize,
        temperature: f32,
        device: &Device,
    ) -> GameHistory {
        // TODO: Load model from most recent checkpoint
        let mut states = Vec::new();

        let (queue_tx, queue_rx) = mpsc::channel();

        let mut mcts = MCTS::<NUM_WORKERS>::new(queue_tx.clone(), device);

        thread::scope(|s| {
            s.spawn(move || evaluation_thread(&self.vit, queue_rx));

            while mcts.board().winner().is_none() {
                mcts.run_simulations(sims_per_move, queue_tx.clone(), device);

                states.push(GameState {
                    board: mcts.board().clone(),
                    distribution: mcts.get_distribution(1.0),
                });

                // TODO: temperature sampler
                let action = mcts.sample_action(temperature);
                mcts.make_action(&action, queue_tx.clone(), device);
            }

            drop(queue_tx);
        });

        GameHistory {
            states,
            winner: mcts.board().winner().unwrap(),
        }
    }
}
