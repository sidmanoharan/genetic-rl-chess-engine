# ♟️ Genetic RL Chess Engine

A self-learning chess engine built from scratch using **Reinforcement Learning** and **Genetic Algorithms** — inspired by DeepMind's AlphaZero. Agents start knowing only the rules and improve purely through self-play.

---

## 🧠 How It Works

Unlike traditional chess engines (Stockfish, etc.) that rely on hand-crafted evaluation functions, this engine:
1. Starts with **zero chess knowledge** beyond the rules
2. Agents play games against each other (self-play)
3. **PPO** trains each agent's neural network to predict better moves
4. **Genetic Algorithms** evolve hyperparameters and network architectures across generations
5. **MCTS** is applied at inference time for stronger play without retraining
6. Only the strongest agents survive and reproduce

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Training Loop                      │
│                                                     │
│  ┌─────────────┐    self-play     ┌─────────────┐  │
│  │  Agent (RL) │ ◄──────────────► │  Agent (RL) │  │
│  │  PPO Policy │                  │  PPO Policy │  │
│  └──────┬──────┘                  └──────┬──────┘  │
│         │  game results                  │          │
│         └──────────────┬─────────────────┘          │
│                        ▼                            │
│              ┌─────────────────┐                    │
│              │ Genetic Engine  │                    │
│              │ - selection     │                    │
│              │ - crossover     │                    │
│              │ - mutation      │                    │
│              └─────────────────┘                    │
│                        │                            │
│                 next generation                     │
└─────────────────────────────────────────────────────┘
```

### Tech Stack
- **Python 3.10+**
- **PyTorch** — neural network policy/value heads
- **python-chess** — rules engine (move generation, legality)
- **NumPy** — board state representation
- **Matplotlib** — training visualisation

---

## 📁 Project Structure

```
genetic-rl-chess-engine/
├── chess_engine/
│   ├── board.py            # Board state, encoding, legal moves
│   └── game.py             # Game loop, result detection
├── genetic/
│   ├── genome.py           # Agent genome (hyperparams + architecture)
│   ├── population.py       # Population management
│   └── operators.py        # Selection, crossover, mutation
├── rl_agent/
│   ├── network.py          # Neural network (policy + value heads)
│   ├── agent.py            # PPO agent
│   ├── memory.py           # Experience replay buffer
│   └── mcts.py             # Monte Carlo Tree Search
├── training/
│   ├── self_play.py        # Self-play game generation
│   ├── trainer.py          # RL training loop
│   └── evolution.py        # GA evolution loop
├── utils/
│   ├── config.py           # All hyperparameters in one place
│   ├── logger.py           # Logging and metrics
│   └── visualiser.py       # Training curves and board display
├── tests/
│   ├── test_board.py
│   ├── test_genetic.py
│   └── test_agent.py
├── scripts/
│   ├── train.py            # Main training entry point
│   └── evaluate.py         # Evaluate a saved agent
├── notebooks/
│   ├── kaggle_train.ipynb  # Kaggle training notebook (3-stage curriculum)
│   └── analysis.ipynb      # ELO curves, game analysis
├── results/
│   ├── training_curves.png # ELO progression across generations
│   └── game.gif            # Trained agent playing itself
├── checkpoints/
│   └── FINAL_best_agent.pt # Trained model weights
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/genetic-rl-chess-engine.git
cd genetic-rl-chess-engine
pip install -r requirements.txt
```

### 2. Run training

```bash
# Start training from scratch
python scripts/train.py

# Resume from a checkpoint
python scripts/train.py --checkpoint checkpoints/gen_0010.pt

# Custom settings
python scripts/train.py --generations 50 --population 20
```

### 3. Evaluate a trained agent

```bash
python scripts/evaluate.py --agent checkpoints/FINAL_best_agent.pt --games 100
```

---

## 📊 Training Progress

The engine tracks:
- **ELO rating** of each agent over time
- **Win/loss/draw rates** per generation
- **Policy entropy** (measures exploration vs exploitation)
- **Value loss** (how well the agent predicts game outcomes)

---

## 🧬 Genetic Algorithm Details

Each agent has a **genome** encoding:

| Gene | Description | Range |
|------|-------------|-------|
| `learning_rate` | PPO learning rate | 1e-5 to 1e-2 |
| `num_filters` | Conv layer width | 64 to 256 |
| `num_residual_blocks` | Network depth | 2 to 8 |
| `gamma` | RL discount factor | 0.9 to 0.999 |
| `entropy_coef` | Exploration bonus | 0.001 to 0.1 |
| `clip_epsilon` | PPO clip range | 0.1 to 0.4 |

**Fitness** = ELO rating after N self-play games against the current population.

---

## 🤖 RL Details

- **Algorithm**: PPO (Proximal Policy Optimisation)
- **State**: 8×8×17 tensor (piece positions, turn, castling rights, en passant)
- **Action**: One of 4672 possible moves (UCI encoded)
- **Reward**: +1 win, −1 loss, 0 draw, small penalty per move to encourage decisive play

---

## 🌲 MCTS Details

Monte Carlo Tree Search is applied at inference time on top of the trained network, with no retraining required. It uses the policy head as move priors and the value head to evaluate leaf nodes, adding approximately 200–300 ELO over the raw network.

```python
from rl_agent.agent import ChessAgent
from rl_agent.mcts import MCTS

agent = ChessAgent()
agent.load("checkpoints/FINAL_best_agent.pt")
mcts = MCTS(agent.network, num_simulations=200)
move = mcts.select_action(board)
```

---

## 📖 Key Papers
- [Mastering Chess with General Reinforcement Learning — AlphaZero](https://arxiv.org/abs/1712.01815)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

## 🗺️ Roadmap
- [x] Chess rules engine
- [x] Board state encoding (17-plane tensor)
- [x] PPO agent with GAE
- [x] Genetic evolution loop
- [x] Self-play training
- [x] MCTS integration
- [ ] Opening book from self-play data
- [ ] Web UI to play against trained agent

---

## ⚠️ Compute Note
Training is GPU-intensive. A full 3-stage curriculum run (70 generations, population up to 20) takes approximately 24 hours on a Kaggle P100. The `notebooks/kaggle_train.ipynb` notebook handles this end-to-end.

---

*Built as a portfolio project demonstrating reinforcement learning, evolutionary algorithms, and systems design.*
