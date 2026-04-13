use std::error::Error;

use alpha_zero_othello::{
    board::{Player, Winner},
    model::vit::ViTConfig,
    train::self_play::AlphaZeroSelfPlay,
};
use candle_core::Device;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::Cpu;

    let self_play: AlphaZeroSelfPlay<4> = AlphaZeroSelfPlay::new(
        ViTConfig {
            d_model: 64,
            n_heads: 4,
            n_layers: 4,
            patch_size: 1,
        },
        &device,
    )?;

    let history = self_play.generate_history(1_000, 1.0, &device);

    match history.winner() {
        Winner::Tie => println!("Tie!"),
        Winner::Player(Player::Black) => println!("Black Won!"),
        Winner::Player(Player::White) => println!("White Won!"),
    }

    for state in history.iter() {
        state.board().display();
    }

    Ok(())
}
