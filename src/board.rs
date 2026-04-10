#[derive(Clone, Copy)]
pub enum Player {
    Black,
    White,
}

#[derive(Clone, Copy)]
pub enum Winner {
    Tie,
    Player(Player),
}

const GRID_SIZE: usize = 8;

#[derive(Clone)]
pub struct Board {
    grid: [[Option<Player>; GRID_SIZE]; GRID_SIZE],
    player: Player,
}

#[derive(Clone, Copy)]
pub struct Action(usize, usize);

impl Board {
    pub fn new() -> Self {
        Self {
            grid: [
                [None; 8],
                [None; 8],
                [None; 8],
                [
                    None,
                    None,
                    None,
                    Some(Player::White),
                    Some(Player::Black),
                    None,
                    None,
                    None,
                ],
                [
                    None,
                    None,
                    None,
                    Some(Player::Black),
                    Some(Player::White),
                    None,
                    None,
                    None,
                ],
                [None; 8],
                [None; 8],
                [None; 8],
            ],
            player: Player::Black,
        }
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        todo!()
    }

    pub fn player(&self) -> Player {
        todo!()
    }

    pub fn winner(&self) -> Option<Winner> {
        todo!()
    }

    pub fn make_action(&mut self, action: &Action) {
        todo!()
    }
}
