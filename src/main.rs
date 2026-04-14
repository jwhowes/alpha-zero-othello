use std::{error::Error, fs, sync::mpsc, thread};

use alpha_zero_othello::{
    board::{Board, Player, Winner},
    model::{
        evaluate_board,
        queue::evaluation_thread,
        vit::{ViT, ViTConfig},
    },
    train::self_play::AlphaZeroSelfPlay,
};
use candle_core::Device;

fn main() -> Result<(), Box<dyn Error>> {
    let config: ViTConfig =
        serde_yaml::from_str(fs::read_to_string("configs/init.yaml")?.as_str())?;

    let device = Device::Cpu;

    let self_play = AlphaZeroSelfPlay::<4>::new(config, &device)?;

    let history = self_play.generate_history(1_000, 1.0, &device);

    match history.winner() {
        Winner::Tie => println!("Tie"),
        Winner::Player(Player::Black) => println!("Black wins!"),
        Winner::Player(Player::White) => println!("White wins!"),
    };

    for state in history.iter() {
        state.board().display();
    }

    Ok(())
}
