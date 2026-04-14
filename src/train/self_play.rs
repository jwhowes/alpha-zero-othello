use std::{sync::mpsc, thread};

use candle_core::{Device, Result};

use crate::{
    mcts::MCTS,
    model::{
        queue::evaluation_thread,
        unet::{UNet, UNetConfig},
        vit::{ViT, ViTConfig},
    },
    train::data::{GameHistory, GameState},
};

pub struct AlphaZeroSelfPlay<const NUM_WORKERS: usize> {
    model: UNet,
}

impl<const NUM_WORKERS: usize> AlphaZeroSelfPlay<NUM_WORKERS> {
    pub fn new(model_config: UNetConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            model: UNet::from_config(model_config, device)?,
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

        let winner = thread::scope(|s| {
            let (queue_tx, queue_rx) = mpsc::channel();

            s.spawn(move || evaluation_thread(&self.model, queue_rx));

            let mut mcts = MCTS::<NUM_WORKERS>::new(queue_tx.clone(), device);

            while mcts.board().winner().is_none() {
                mcts.run_simulations(sims_per_move, queue_tx.clone(), device);
                println!("Hello!");

                states.push(GameState {
                    board: mcts.board().clone(),
                    distribution: mcts.get_distribution(1.0),
                });

                // TODO: temperature sampler
                let action = mcts.sample_action(temperature);
                mcts.make_action(&action, queue_tx.clone(), device);
            }

            mcts.board().winner().unwrap()
        });

        GameHistory { states, winner }
    }
}
