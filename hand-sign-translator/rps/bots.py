import random
from collections import Counter
from typing import List

MOVES = ["rock", "paper", "scissors"]

class RandomBot:
    def predict(self, history: List[str]) -> str:
        return random.choice(MOVES)

class FrequencyBot:
    def predict(self, history: List[str]) -> str:
        if not history:
            return random.choice(MOVES)
        
        # Ensure all moves in history are lowercase
        history = [move.lower() for move in history]
        
        freq = Counter(history)
        most_common = freq.most_common(1)[0][0]
        return MOVES[(MOVES.index(most_common) + 1) % 3]

    def __str__(self):
        return "FrequencyBot"