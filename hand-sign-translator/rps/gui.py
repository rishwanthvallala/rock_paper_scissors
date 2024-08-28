import tkinter as tk
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GameGUI:
    def __init__(self, master):
        self.master = master
        master.title("Rock Paper Scissors")

        # Frame for the camera feed
        self.camera_frame = tk.Frame(master)
        self.camera_frame.pack(side=tk.LEFT)

        self.camera_canvas = tk.Canvas(self.camera_frame, width=640, height=480)
        self.camera_canvas.pack()

        # Frame for game info and controls
        self.info_frame = tk.Frame(master)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Game counters
        self.counters_frame = tk.Frame(self.info_frame)
        self.counters_frame.pack(pady=10)

        self.human_wins_label = tk.Label(self.counters_frame, text="Human Wins: 0", font=("Arial", 14))
        self.human_wins_label.grid(row=0, column=0, padx=10)

        self.ai_wins_label = tk.Label(self.counters_frame, text="AI Wins: 0", font=("Arial", 14))
        self.ai_wins_label.grid(row=0, column=1, padx=10)

        self.ties_label = tk.Label(self.counters_frame, text="Ties: 0", font=("Arial", 14))
        self.ties_label.grid(row=0, column=2, padx=10)

        self.bot_label = tk.Label(self.info_frame, text="Current Bot: None", font=("Arial", 14))
        self.bot_label.pack()

        self.stats_text = tk.Text(self.info_frame, height=10, width=40, font=("Arial", 12))
        self.stats_text.pack()

        self.result_label = tk.Label(self.info_frame, text="", font=("Arial", 16))
        self.result_label.pack()

        self.next_round_button = tk.Button(self.info_frame, text="Next Round", command=self.next_round, font=("Arial", 14))
        self.next_round_button.pack(pady=10)

        # Performance graph
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.info_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.camera_canvas.image = photo

    def update_result(self, result):
        self.result_label.config(text=result)

    def update_bot(self, bot_name, bot_stats):
        self.bot_label.config(text=f"Current Bot: {bot_name}")
        if bot_stats:
            stats_text = f"Bot: {bot_name}\n"
            stats_text += f"Total Score: {bot_stats['total_score']:.2f}\n"
            stats_text += f"Total Pulls: {bot_stats['total_pulls']}\n"
            stats_text += f"Average Score: {bot_stats['average_score']:.2f}\n"
            stats_text += f"UCB Value: {bot_stats['ucb_value']:.2f}\n"
            stats_text += f"Wins: {bot_stats['wins']}\n"
            stats_text += f"Losses: {bot_stats['losses']}\n"
            stats_text += f"Ties: {bot_stats['ties']}\n"
            stats_text += f"Win Rate: {bot_stats['win_rate']:.2%}\n"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_text)

    def update_performance_graph(self, recent_performance):
        self.ax.clear()
        self.ax.plot(recent_performance)
        self.ax.set_title("Recent Performance")
        self.ax.set_xlabel("Round")
        self.ax.set_ylabel("Score")
        self.canvas.draw()

    def update_counters(self, human_wins, ai_wins, ties):
        self.human_wins_label.config(text=f"Human Wins: {human_wins}")
        self.ai_wins_label.config(text=f"AI Wins: {ai_wins}")
        self.ties_label.config(text=f"Ties: {ties}")

    def next_round(self):
        self.master.event_generate("<<NextRound>>")