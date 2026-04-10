# AlphaZero Othello

A deep learning reinforcement learning program for the game of (Othello)[https://en.wikipedia.org/wiki/Reversi].

The training routine is based on (AlphaGo Zero)[https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf].

## Self Play

The neural network $f_\theta$ takes as input a board state $b$ and produces a prior distribution $\pi(a | b)$ over the legal moves $a \in \mathcal{A}(b)$ and a value $v_b \in [-1, 1]$.

During self play, a sub tree of the current game tree is maintained and updated. For each edge $(b, a)$ in this tree we store:
1. The total visit count of the edge, $N(b, a)$
2. The prior value of the edge, $\pi(a | b)$
3. The total action value of the edge's descendants, $Q(b, a)$

During self play, the model performs $S$ simulations per action. Each simulation adds a node to the game tree. To add a node, we begin from the root $b_0$ and iteratively traverse through $b_1, b_2$ and so on until we reach a leaf $b_T$ at step $T$.

Given $b_t$, the state $b_{t+1} | b_t, a_t$ is selected by choosing the edge $(b_t, a_t)$ which maximises
$$
W(b_t, a_t) = \frac{Q(b_t, a_t)}{N(b_t, a_t)} + c \pi(a_t | b_t) \frac{\sqrt{\sum_{a_{t-1}}N(b_{t-1}, a_{t-1})}}{1 + N(b_t, a_t)}
$$
where $c > 0$ is a hyperparameter controlling the balance of exploration vs. exploitation.

An exception to the above is if if $N(b_t, a_t) = 0$ (i.e. the edge is unexplored) in which case we replace $\frac{Q(b_t, a_t)}{N(b_t, a_t)}$ with $0$.

When we reach a leaf $b_T$, we use the neural network to compute $f_\theta(b_T) = (\pi(a | b_T), v_{b_T})$ and we initialise the edges $(b_T, a)$ such that:
1. $N(b_T, a) = 0$
2. The prior $\pi$ is set to the network's output
3. $Q(b_T, a) = 0$

We then backtrack up the tree, updating each $(b_t, a_t)$ like so:
1. $N(b_t, a_t) \rightarrow N(b_t, a_t + 1)$
2. $Q(b_t, a_t) \rightarrow Q(b_t, a_t) + v_{b_T}$ if $b_t$ is white to move, otherwise $Q(b_t, a_t) \rightarrow Q(b_t, a_t) - v_{b_T}$

Note that the network $f_\theta$ always produces its valuation from the perspective of white, hence the need for negating the value from black's perspective.

After all $S$ simulations have been run, the action $a$ is selected based on the probability distribution
$$
p(a | b_0) = \frac{N(b_0, a)^{1 / \tau}}{\sum_{a'} N(b_0, a')^{1/\tau}}
$$
where $\tau > 0$ is the temperature.

Note that the already explored descendants of $b_1$ are kept in the tree to reduce unnecessary re-exploration.

## Training

For each game of self play, we store:
1. The sequence of game states and actions $(b_0, a_0), ..., (b_{T-1}, a_{T-1})$
2. The game's value $v^*$ which is set to $1$ if white won, $-1$ if black won and $0$ for a tie
3. The distribution $p^*(a | b_t) = \frac{N(b_t, a)}{\sum_{a'} N(b_t, a')}$ for each game state $b_t$

Given state $b_t$ the model's output $f_\theta(b_t) = (\pi(a | b_t), v_{b_t})$ is trained to minimise
$$
\mathcal{L}(b_t) = ||v^* - v_{b_t}||_2^2 - \lambda\pi\log p^*
$$
where $\lambda > 0$ is a scaling factor.