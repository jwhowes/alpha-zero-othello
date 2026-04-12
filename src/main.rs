use std::{
    error::Error,
    io::{Write, stdin, stdout},
};

use alpha_zero_othello::board::{Board, action::Action};

fn main() -> Result<(), Box<dyn Error>> {
    let mut board = Board::new();

    while board.winner().is_none() {
        let mut buf = String::new();

        board.display();

        print!("Enter your move: ");
        stdout().flush()?;

        stdin().read_line(&mut buf)?;

        let action: Action = buf.trim().parse()?;

        println!("{:?}", action);

        board.make_action(&action);
    }

    Ok(())
}
