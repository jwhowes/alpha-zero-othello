use std::{error::Error, fmt::Display, str::FromStr};

use crate::board::GRID_SIZE;

#[derive(Debug, Clone, Copy)]
pub struct Action(pub usize, pub usize);

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}",
            char::from_u32(self.0 as u32 + 65).unwrap(),
            self.1
        )
    }
}

#[derive(Debug)]
pub enum ActionParseError {
    OutOfBounds,
    SyntaxError,
}

impl Display for ActionParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::OutOfBounds => "Index out of bounds",
            Self::SyntaxError => "Invalid syntax",
        };

        write!(f, "{}", s)
    }
}

impl Error for ActionParseError {}

impl FromStr for Action {
    type Err = ActionParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 2 {
            Err(ActionParseError::SyntaxError)
        } else {
            let bytes = s.as_bytes();

            let col = bytes[0] as isize - 65;
            let row = bytes[1] as isize - 49;

            if col < 0 || row < 0 {
                Err(ActionParseError::SyntaxError)
            } else if col > GRID_SIZE as isize || row > GRID_SIZE as isize {
                Err(ActionParseError::OutOfBounds)
            } else {
                Ok(Self(col as usize, row as usize))
            }
        }
    }
}
