#[derive(Clone, Copy, PartialEq)]
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

const DIRECTIONS: [(isize, isize); 6] = [(1, -1), (1, 0), (1, 1), (-1, -1), (-1, 0), (-1, 1)];

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
        let mut legal = Vec::new();

        for x in 0..GRID_SIZE {
            for y in 0..GRID_SIZE {
                let action = Action(x, y);
                if !self.legal_directions(&action).is_empty() {
                    legal.push(action);
                }
            }
        }

        legal
    }

    pub fn player(&self) -> Player {
        self.player
    }

    pub fn winner(&self) -> Option<Winner> {
        if self.legal_actions().is_empty() {
            let mut black_count = 0;
            let mut white_count = 0;

            for x in 0..GRID_SIZE {
                for y in 0..GRID_SIZE {
                    match self.grid[y][x] {
                        None => {}
                        Some(Player::Black) => black_count += 1,
                        Some(Player::White) => white_count += 1,
                    }
                }
            }

            if black_count == white_count {
                Some(Winner::Tie)
            } else if black_count > white_count {
                Some(Winner::Player(Player::Black))
            } else {
                Some(Winner::Player(Player::White))
            }
        } else {
            None
        }
    }

    fn legal_directions(&self, Action(x, y): &Action) -> Vec<usize> {
        let player = self.player();
        let opponent = match player {
            Player::White => Player::Black,
            Player::Black => Player::White,
        };

        let mut legal = Vec::with_capacity(6);

        let (x, y) = (*x as isize, *y as isize);

        for (i, (dx, dy)) in DIRECTIONS.iter().enumerate() {
            if x + dx >= 0
                && x + dx < GRID_SIZE as isize
                && y + dy >= 0
                && y + dy < GRID_SIZE as isize
                && self.grid[(y + dy) as usize][(x + dx) as usize] == Some(opponent)
            {
                for n in 2..GRID_SIZE as isize {
                    if x + n * dx >= 0
                        && x + n * dx < GRID_SIZE as isize
                        && y + n * dy >= 0
                        && y + n * dy < GRID_SIZE as isize
                    {
                        match self.grid[(y + n * dy) as usize][(x + n * dx) as usize] {
                            None => {
                                break;
                            }
                            Some(cell_player) => {
                                if cell_player == player {
                                    legal.push(i);
                                    break;
                                }
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        legal
    }

    pub fn make_action(&mut self, action: &Action) {
        let player = self.player;

        let opponent = match self.player {
            Player::Black => Player::White,
            Player::White => Player::Black,
        };

        let (x, y) = (action.0 as isize, action.1 as isize);

        for i in self.legal_directions(action).into_iter() {
            let (dx, dy) = DIRECTIONS[i];
            let mut n = 1;

            while self.grid[(y + n * dy) as usize][(x + n * dx) as usize] == Some(opponent) {
                self.grid[(y + n * dy) as usize][(x + n * dx) as usize] = Some(self.player);
                n += 1;
            }
        }

        self.player = opponent;
        if self.legal_actions().is_empty() {
            self.player = player;
        }
    }
}
