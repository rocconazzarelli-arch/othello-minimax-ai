# Othello Minimax AI

An Othello/Reversi agent built with **minimax search**, **alpha-beta pruning**, move ordering, and a learned positional evaluation table.

## Strategy

The agent evaluates legal moves from its own perspective and searches the game tree using:

- depth-limited minimax;
- alpha-beta pruning;
- one-ply move ordering;
- deeper search in the endgame;
- a symmetric board-position weight table;
- a bounded `tanh` evaluation score.

The search depth is four plies during the middle game and six plies when ten or fewer squares remain empty.

## Package interface

The module exposes the course-compatible factory function:

```python
from othello_agent import cria_jogador

agent = cria_jogador()
```

The game engine must provide `jogadas_legais()`, `joga(move)`, and `posicao((row, column))` methods. The agent itself is independent of a particular UI.

## Repository structure

```text
.
├── src/othello_agent/
│   ├── __init__.py
│   ├── player.py
│   └── weights.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Install for development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Formal tournament win-rate results were not available in the original project folder, so this repository does not claim a benchmark score.

## Author

Rocco Nazzarelli

M.Sc. Computer Science and Engineering – Artificial Intelligence, Politecnico di Milano
