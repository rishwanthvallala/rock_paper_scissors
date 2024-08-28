import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import random
from typing import List, Tuple
import tkinter as tk
from PIL import Image, ImageTk

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
# Custom layer to handle unknown configurations
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)
        
        

class CameraInterface:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.imgSize = 300
        self.offset = 20
        
        # Load the model
        try:
            with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
                self.model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
        except Exception as e:
            print(f"Failed to load Keras model: {str(e)}")
            print("Trying TFLite conversion...")
            model = tf.keras.models.load_model("Model/keras_model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open('model.tflite', 'wb') as f:
                f.write(tflite_model)
            self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        with open("Model/labels.txt", "r") as file:
            self.labels = [line.strip() for line in file.readlines()]

    def get_frame(self):
        success, img = self.cap.read()
        if not success:
            return None
        
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                wGap = math.ceil((self.imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = self.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                hGap = math.ceil((self.imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = self.get_prediction(imgWhite)

            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset-50),
                          (x - self.offset+90, y - self.offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, self.labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-self.offset, y-self.offset),
                          (x + w+self.offset, y + h+self.offset), (255, 0, 255), 4)

        return imgOutput, self.labels[index] if hands else None

    def get_prediction(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        if hasattr(self, 'interpreter'):
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            prediction = self.model.predict(img)
        
        index = np.argmax(prediction)
        return prediction[0], index

    def release(self):
        self.cap.release()

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
        self.master.event_generate("<<NextRound>>")

def play_game(camera: CameraInterface, gui: GameGUI, bandit: MultiArmedBandit, bots: List):
    human_move = None
    
    def update():
        nonlocal human_move
        frame, detected_move = camera.get_frame()
        if frame is not None:
            gui.update_frame(frame)
        if detected_move is not None:
            human_move = detected_move.lower()
        gui.master.after(10, update)

    def on_next_round(event):
        nonlocal human_move
        if human_move is None:
            gui.update_result("No move detected. Try again.")
            return

        bot_index = bandit.choose_bot()
        bot_move = bots[bot_index].predict([])  # Empty history for simplicity

        result = play_rps(bot_move, human_move)

        if result == 1:
            gui.update_result(f"AI wins! {bot_move.capitalize()} beats {human_move}")
        elif result == -1:
            gui.update_result(f"Human wins! {human_move.capitalize()} beats {bot_move}")
        else:
            gui.update_result(f"It's a tie! Both chose {human_move}")

        bandit.update(bot_index, result)
        human_move = None

    gui.master.bind("<<NextRound>>", on_next_round)
    gui.master.after(10, update)

if __name__ == "__main__":
    bots = [RandomBot(), FrequencyBot()]
    bandit = MultiArmedBandit(bots)

    root = tk.Tk()
    gui = GameGUI(root)
    camera = CameraInterface()

    play_game(camera, gui, bandit, bots)

    root.mainloop()
    camera.release()