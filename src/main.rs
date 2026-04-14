use std::{error::Error, fs, sync::mpsc, thread};

use alpha_zero_othello::{
    board::Board,
    model::{
        evaluate_board,
        queue::evaluation_thread,
        vit::{ViT, ViTConfig},
    },
};
use candle_core::Device;

fn main() -> Result<(), Box<dyn Error>> {
    let config: ViTConfig =
        serde_yaml::from_str(fs::read_to_string("configs/init.yaml")?.as_str())?;

    let device = Device::Cpu;

    let model = ViT::from_config(config, &device)?;

    let (queue_tx, queue_rx) = mpsc::channel();

    thread::scope(|s| {
        s.spawn(move || evaluation_thread(&model, queue_rx));

        let board = Board::new();

        let (_prior, value) = evaluate_board(&board, queue_tx.clone(), &device).unwrap();

        println!("{}", value);

        drop(queue_tx);
    });

    Ok(())
}
