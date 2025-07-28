# Auction Gym

A simulation environment ("gym") for training agents in a Real-Time Bidding (RTB) auction environment.

## Overview
This repository provides a modular, extensible environment for simulating RTB auctions, enabling the training and evaluation of reinforcement learning agents and other strategies in a realistic auction setting.

## Features
- RTB auction simulation (e.g., second-price auction)
- Customizable agent interfaces
- Gym-like API for easy integration with RL libraries
- Extensible for different auction types and agent strategies

## Project Structure
```
auction_gym/
│
├── auction_gym/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── rtb_env.py
│   └── core/
│       ├── __init__.py
│       ├── auction.py
│       ├── agent.py
│       └── utils.py
├── tests/
│   └── test_rtb_env.py
├── setup.py
├── README.md
└── requirements.txt
```

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   pytest
   ```

## License
MIT 