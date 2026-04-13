use std::{
    error::Error,
    io::{Write, stdin, stdout},
};

use alpha_zero_othello::{
    board::{Player, action::Action},
    mcts::MCTS,
};

const SIMS_PER_MOVE: usize = 1_000;
const NUM_WORKERS: usize = 4;

fn main() -> Result<(), Box<dyn Error>> {
    let mut mcts: MCTS<NUM_WORKERS> = MCTS::new();

    while mcts.board().winner().is_none() {
        let action = match mcts.board().player() {
            Player::Black => {
                mcts.board().display();

                let mut buf = String::new();

                print!("Enter your move: ");

                let _ = stdout().flush()?;

                let _ = stdin().read_line(&mut buf)?;

                buf.trim().parse::<Action>()?
            }

            _ => {
                mcts.run_simulations(SIMS_PER_MOVE);

                let action = mcts.sample_action(1.0);

                println!("Computer move: {}", &action);

                action
            }
        };

        mcts.make_action(&action);
    }

    Ok(())
}
