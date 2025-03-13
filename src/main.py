from utils.load_config import LoadConfig
from chatbot import Chatbot

if __name__ == "__main__":
    config = LoadConfig()
    bot = Chatbot(config)

    print("✅ Chatbot is ready! Type 'q' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "q":
            break

        response = bot.chat(user_input)
        print("\nBot:", response)
