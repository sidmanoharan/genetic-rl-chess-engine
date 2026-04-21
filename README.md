# ♟️ Chess RL Engine

A self-learning chess engine built from scratch using **Reinforcement Learning** and **Genetic Algorithms** — inspired by DeepMind's AlphaZero. Agents start knowing only the rules of chess and improve exclusively by playing against each other.

---

## 🧠 How It Works

### The Core Idea
Unlike traditional chess engines (Stockfish, etc.) that rely on hand-crafted evaluation functions, this engine:
1. Starts with **zero chess knowledge** beyond the rules
2. Agents play games against each other (self-play)
3. **RL** (PPO) trains the neural network policy to predict good moves
4. **Genetic Algorithms** evolve hyperparameters and network architectures across generations
5. Only the strongest agents survive and reproduce

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
- **Matplotlib / Plotly** — training visualisation

---

## 📁 Project Structure

```
chess-rl-engine/
├── chess_engine/
│   ├── board.py          # Board state, encoding, legal moves
│   └── game.py           # Game loop, result detection
├── genetic/
│   ├── genome.py         # Agent genome (hyperparams + architecture)
│   ├── population.py     # Population management
│   └── operators.py      # Selection, crossover, mutation
├── rl_agent/
│   ├── network.py        # Neural network (policy + value heads)
│   ├── agent.py          # PPO agent
│   └── memory.py         # Experience replay buffer
├── training/
│   ├── self_play.py      # Self-play game generation
│   ├── trainer.py        # RL training loop
│   └── evolution.py      # GA evolution loop
├── utils/
│   ├── config.py         # All hyperparameters in one place
│   ├── logger.py         # Logging and metrics
│   └── visualiser.py     # Training curves and board display
├── tests/
│   ├── test_board.py
│   ├── test_genetic.py
│   └── test_agent.py
├── scripts/
│   ├── train.py          # Main training entry point
│   └── evaluate.py       # Evaluate a saved agent
├── notebooks/
│   └── analysis.ipynb    # ELO curves, game analysis
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/chess-rl-engine.git
cd chess-rl-engine
pip install -r requirements.txt
```

### 2. Run training

```bash
# Start training from scratch
python scripts/train.py

# Resume from a checkpoint
python scripts/train.py --checkpoint checkpoints/gen_10.pt

# Custom config
python scripts/train.py --generations 50 --population 20
```

### 3. Evaluate an agent

```bash
python scripts/evaluate.py --agent checkpoints/best_agent.pt --games 100
```

---

## 📊 Training Progress

The engine tracks:
- **ELO rating** of each agent over time
- **Win/loss/draw rates** per generation
- **Policy entropy** (measures exploration vs exploitation)
- **Value loss** (how well the agent predicts outcomes)

---

## 🧬 Genetic Algorithm Details

Each agent has a **genome** encoding:
| Gene | Description | Range |
|------|-------------|-------|
| `learning_rate` | PPO learning rate | 1e-5 to 1e-2 |
| `hidden_size` | Network hidden layer size | 64 to 512 |
| `n_layers` | Number of residual blocks | 2 to 10 |
| `gamma` | RL discount factor | 0.9 to 0.999 |
| `entropy_coef` | Exploration bonus | 0.001 to 0.1 |
| `clip_epsilon` | PPO clip range | 0.1 to 0.4 |

**Fitness** = ELO rating after N self-play games against the current population.

---

## 🤖 RL Details

- **Algorithm**: PPO (Proximal Policy Optimisation)
- **State**: 8×8×17 tensor (piece positions, turn, castling rights, en passant)
- **Action**: One of 4672 possible moves (all legal UCI moves encoded)
- **Reward**: +1 win, -1 loss, 0 draw, small negative per move (encourages decisive play)

---

## 📖 Key Papers
- [Mastering Chess with General Reinforcement Learning (AlphaZero)](https://arxiv.org/abs/1712.01815)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

## 🗺️ Roadmap
- [x] Chess rules engine
- [x] Board state encoding
- [x] PPO agent
- [x] Genetic evolution loop
- [x] Self-play training
- [ ] MCTS (Monte Carlo Tree Search) integration
- [ ] Opening book from self-play data
- [ ] Web UI to play against trained agent

---

## ⚠️ Compute Warning
Training is computationally intensive. Recommended: GPU with 8GB+ VRAM, or use Google Colab. A full training run (50 generations, 20 agents) takes ~12 hours on a modern GPU.

---

*Built as a portfolio project demonstrating RL, evolutionary algorithms, and systems design.*
