#[derive(Clone, Copy)]
pub enum Player {
    PlayerOne,
    PlayerTwo,
}

#[derive(Clone, Copy)]
pub enum Winner {
    Tie,
    Player(Player),
}

#[derive(Clone)]
pub struct Board {
    // TODO
}

#[derive(Clone, Copy)]
pub struct Action(usize, usize);

impl Board {
    pub fn new() -> Self {
        todo!()
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
