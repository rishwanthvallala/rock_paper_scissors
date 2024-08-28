import numpy as np
from typing import List, Dict
import math

class MultiArmedBandit:
    def __init__(self, bots: List, c: float = 2.0):
        self.bots = bots
        self.n_bots = len(bots)
        self.c = c  # Exploration parameter for UCB
        self.total_pulls = 0
        self.scores = np.zeros(self.n_bots)
        self.pulls = np.zeros(self.n_bots)
        self.ucb_values = np.zeros(self.n_bots)
        self.recent_performance = []  # Track recent performance
        self.wins = np.zeros(self.n_bots)
        self.losses = np.zeros(self.n_bots)
        self.ties = np.zeros(self.n_bots)

    def choose_bot(self) -> int:
        self.total_pulls += 1
        if self.total_pulls <= self.n_bots:
            return self.total_pulls - 1  # Ensure each bot is tried at least once
        
        # Calculate UCB values
        exploration = np.sqrt((2 * np.log(self.total_pulls)) / self.pulls)
        self.ucb_values = (self.scores / self.pulls) + (self.c * exploration)
        
        return np.argmax(self.ucb_values)

    def update_all(self, human_move: str, rewards: List[float]):
        self.pulls += 1
        self.scores += rewards
        self.recent_performance.append((np.argmax(rewards), max(rewards)))
        if len(self.recent_performance) > 100:  # Keep only last 100 rounds
            self.recent_performance.pop(0)
        
        for i, reward in enumerate(rewards):
            if reward == 1:
                self.wins[i] += 1
            elif reward == -1:
                self.losses[i] += 1
            else:
                self.ties[i] += 1

    def get_best_bot(self) -> int:
        return np.argmax(self.scores / np.maximum(self.pulls, 1))

    def get_bot_stats(self) -> List[Dict]:
        stats = []
        for i in range(self.n_bots):
            total_games = self.wins[i] + self.losses[i] + self.ties[i]
            win_rate = self.wins[i] / total_games if total_games > 0 else 0
            stats.append({
                'bot_name': self.bots[i].__class__.__name__,
                'total_score': self.scores[i],
                'total_pulls': self.pulls[i],
                'average_score': self.scores[i] / max(self.pulls[i], 1),
                'ucb_value': self.ucb_values[i],
                'wins': self.wins[i],
                'losses': self.losses[i],
                'ties': self.ties[i],
                'win_rate': win_rate
            })
        return stats

    def get_recent_performance(self, n: int = 20) -> List[float]:
        if not self.recent_performance:
            return []
        recent_n = self.recent_performance[-n:]
        return [reward for _, reward in recent_n]

    def reset(self):
        self.total_pulls = 0
        self.scores = np.zeros(self.n_bots)
        self.pulls = np.zeros(self.n_bots)
        self.ucb_values = np.zeros(self.n_bots)
        self.recent_performance = []
        self.wins = np.zeros(self.n_bots)
        self.losses = np.zeros(self.n_bots)
        self.ties = np.zeros(self.n_bots)