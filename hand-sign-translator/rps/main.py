import tkinter as tk
from game_logic import play_game
from gui import GameGUI
from camera_interface import CameraInterface
from bots import RandomBot, FrequencyBot
from multi_armed_bandit import MultiArmedBandit

if __name__ == "__main__":
    bots = [RandomBot(), FrequencyBot()]
    bandit = MultiArmedBandit(bots)

    root = tk.Tk()
    gui = GameGUI(root)
    camera = CameraInterface()

    play_game(camera, gui, bandit, bots)

    root.mainloop()
    camera.release()