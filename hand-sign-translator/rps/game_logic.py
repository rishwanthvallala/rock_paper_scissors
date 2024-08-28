from typing import List
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MOVES = ["rock", "paper", "scissors"]

def play_rps(move1: str, move2: str) -> int:
    move1 = move1.split()[-1].lower() if ' ' in move1 else move1.lower()
    move2 = move2.split()[-1].lower() if ' ' in move2 else move2.lower()
    
    if move1 == move2:
        return 0
    elif (MOVES.index(move1) - MOVES.index(move2)) % 3 == 1:
        return 1
    else:
        return -1

def play_game(camera, gui, bandit, bots: List):
    human_move = None
    move_history = []
    human_wins = 0
    ai_wins = 0
    ties = 0
    
    def update():
        nonlocal human_move
        frame, detected_move = camera.get_frame()
        if frame is not None:
            gui.update_frame(frame)
        if detected_move is not None:
            human_move = detected_move.lower()  # Convert to lowercase
        gui.master.after(10, update)

    def on_next_round(event):
        nonlocal human_move, move_history, human_wins, ai_wins, ties
        if human_move is None:
            gui.update_result("No move detected. Try again.")
            return

        chosen_bot_index = bandit.choose_bot()
        all_bot_moves = [bot.predict(move_history) for bot in bots]
        chosen_bot_move = all_bot_moves[chosen_bot_index]

        # Update GUI with current bot and its stats
        chosen_bot_name = bots[chosen_bot_index].__class__.__name__
        chosen_bot_stats = bandit.get_bot_stats()[chosen_bot_index] if bandit.get_bot_stats() else None
        gui.update_bot(chosen_bot_name, chosen_bot_stats)

        # Print which bot is being used (console)
        print(f"Using {chosen_bot_name} this round")
        if chosen_bot_stats:
            print(f"Average score: {chosen_bot_stats['average_score']:.2f}")

        result = play_rps(chosen_bot_move, human_move)

        human_move_display = human_move.split()[-1] if ' ' in human_move else human_move

        if result == 1:
            gui.update_result(f"AI wins! {chosen_bot_move.capitalize()} beats {human_move_display.capitalize()}")
            ai_wins += 1
        elif result == -1:
            gui.update_result(f"Human wins! {human_move_display.capitalize()} beats {chosen_bot_move.capitalize()}")
            human_wins += 1
        else:
            gui.update_result(f"It's a tie! Both chose {human_move_display.capitalize()}")
            ties += 1

        # Update move history
        move_history.append(human_move_display.lower())  # Ensure lowercase when adding to history
        if len(move_history) > 10:  # Keep only the last 10 moves
            move_history = move_history[-10:]

        # Calculate rewards for all bots
        all_rewards = [play_rps(bot_move, human_move) for bot_move in all_bot_moves]
        bandit.update_all(human_move, all_rewards)

        human_move = None

        # Update performance graph
        recent_performance = bandit.get_recent_performance()
        gui.update_performance_graph(recent_performance)

        # Update game counters
        gui.update_counters(human_wins, ai_wins, ties)

    gui.master.bind("<<NextRound>>", on_next_round)
    gui.master.after(10, update)