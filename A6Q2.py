import random

class Rock_paper_scissors:
    def __init__(self, total_rounds):
        self.total_rounds = total_rounds
        self.current_round = 1
        self.user_wins = 0
        self.computer_wins = 0
        self.choices = ["rock", "paper", "scissors"]
    
    def get_computer_choice(self):
        return random.choice(self.choices)
    
    def find_winner(self, user_choice, computer_choice):
        if user_choice == computer_choice:
            return "It's a tie!"
        elif ((user_choice == "rock" and computer_choice == "scissors") or 
             (user_choice == "scissors" and computer_choice == "paper") or 
             (user_choice == "paper" and computer_choice == "rock")):
            self.user_wins += 1
            return "You win this round!"
        else:
            self.computer_wins += 1
            return "Computer wins this round!"
    
    def check_game_winner(self):
        if self.user_wins > self.computer_wins:
            return "You win the game!"
        elif self.computer_wins > self.user_wins:
            return "Computer wins the game!"
        else:
            return "The game is a tie!"

    def play_round(self, user_choice):
        if self.current_round > self.total_rounds:
            return "Game over. The game has already finished."

        computer_choice = self.get_computer_choice()

        round_result = self.find_winner(user_choice, computer_choice)
        
        self.current_round += 1
        
        print(f"Round {self.current_round - 1}:")
        print(f"Your choice: {user_choice}")
        print(f"Computer's choice: {computer_choice}")
        print(round_result)
        print(f"Score - You: {self.user_wins}, Computer: {self.computer_wins}")
        
        if self.current_round > self.total_rounds:
            print(self.check_game_winner())
        
game = Rock_paper_scissors(5)
    
while game.current_round <= game.total_rounds:
        user_input = input("Enter your choice (rock, paper, or scissors): ").lower()
        if user_input in game.choices:
            game.play_round(user_input)
        else:
            print("Invalid choice. Please enter rock, paper, or scissors.")
