import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import random
from typing import Counter, List, Tuple

# Define the possible moves
MOVES = ["rock", "paper", "scissors"]

class RandomBot:
    def predict(self, history: List[str]) -> str:
        return random.choice(MOVES)

class FrequencyBot:
    def predict(self, history: List[str]) -> str:
        if not history:
            return random.choice(MOVES)
        
        freq = Counter(history)
        most_common = freq.most_common(1)[0][0]
        return MOVES[(MOVES.index(most_common) + 1) % 3]

class MultiArmedBandit:
    def __init__(self, bots: List, epsilon: float = 0.1):
        self.bots = bots
        self.epsilon = epsilon
        self.scores = [0] * len(bots)
        self.pulls = [0] * len(bots)

    def choose_bot(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(self.bots) - 1)
        else:
            return max(range(len(self.bots)), key=lambda i: self.scores[i] / (self.pulls[i] + 1))

    def update(self, bot_index: int, reward: float):
        self.scores[bot_index] += reward
        self.pulls[bot_index] += 1

def play_rps(move1: str, move2: str) -> int:
    if move1 == move2:
        return 0
    elif (MOVES.index(move1) - MOVES.index(move2)) % 3 == 1:
        return 1
    else:
        return -1

def simulate(num_rounds: int) -> Tuple[List[int], List[float]]:
    bots = [RandomBot(), FrequencyBot()]
    bandit = MultiArmedBandit(bots)
    history = []
    bot_choices = []
    cumulative_reward = 0
    rewards = []

    for _ in range(num_rounds):
        bot_index = bandit.choose_bot()
        bot_choices.append(bot_index)
        
        bot_move = bots[bot_index].predict(history)
        opponent_move = random.choice(MOVES)
        
        result = play_rps(bot_move, opponent_move)
        bandit.update(bot_index, result)
        
        history.append(opponent_move)
        cumulative_reward += result
        rewards.append(cumulative_reward)

    return bot_choices, rewards
# New class for camera interface
class CameraInterface:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model = load_model('rps_model.h5')  # Assume we have a trained model
        self.classes = ['rock', 'paper', 'scissors']

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def classify_gesture(self, frame):
        resized = cv2.resize(frame, (150, 150))
        normalized = img_to_array(resized) / 255.0
        prediction = self.model.predict(np.expand_dims(normalized, axis=0))[0]
        return self.classes[np.argmax(prediction)]

    def release(self):
        self.cap.release()

# New class for GUI
class GameGUI:
    def __init__(self, master):
        self.master = master
        master.title("Rock Paper Scissors")

        self.canvas = tk.Canvas(master, width=640, height=480)
        self.canvas.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.next_round_button = tk.Button(master, text="Next Round", command=self.next_round)
        self.next_round_button.pack()

    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def update_result(self, result):
        self.result_label.config(text=result)

    def next_round(self):
        # This will be connected to the main game loop
        pass

# Main game loop
def play_game(camera: CameraInterface, gui: GameGUI, bandit: MultiArmedBandit, bots: List):
    while True:
        # Get frame from camera
        frame = camera.get_frame()
        gui.update_frame(frame)

        # Wait for "Next Round" button press
        gui.master.wait_variable(gui.next_round_button)

        # Classify human gesture
        human_move = camera.classify_gesture(frame)

        # Choose bot and get its move
        bot_index = bandit.choose_bot()
        bot_move = bots[bot_index].predict([])  # Empty history for simplicity

        # Determine winner
        result = play_rps(bot_move, human_move)

        # Update GUI and bandit
        if result == 1:
            gui.update_result(f"AI wins! {bot_move.capitalize()} beats {human_move}")
        elif result == -1:
            gui.update_result(f"Human wins! {human_move.capitalize()} beats {bot_move}")
        else:
            gui.update_result(f"It's a tie! Both chose {human_move}")

        bandit.update(bot_index, result)

# Main execution
if __name__ == "__main__":
    camera = CameraInterface()
    bots = [RandomBot(), FrequencyBot()]
    bandit = MultiArmedBandit(bots)

    root = tk.Tk()
    gui = GameGUI(root)

    play_game(camera, gui, bandit, bots)

    root.mainloop()
    camera.release()