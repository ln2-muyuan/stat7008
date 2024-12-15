from nltk.chat.util import Chat, reflections
from function1 import analyze_text_file
from function2 import get_topic_by_filename
from function3 import run_sentiment_analysis
from function4 import function4
from pairs import pairs
from function5.function5 import model_prediction

import os

class ChatBot:
    def __init__(self):
        self.chatbot = Chat(pairs, reflections)

    def list_pdf_files(self, folder_path):
        """List all PDF files in the given folder."""
        return [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]

    def start_conversation(self):
        """Start a daily conversation using nltk.chat.util.Chat."""
        print("Chatbot: Starting the conversation...")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Chatbot: Goodbye!")
                break
            if user_input.lower() == "back":
                print("Chatbot: Returning to the main menu...")
                break
            response = self.chatbot.respond(user_input)
            if response:
                print(f"Chatbot: {response}")
            else:
                print("Chatbot: I don't understand that. Can you rephrase?")

    def handle_pdf_selection(self):
        """List and handle operations on PDF files."""
        folder_path = "extracted_text"
        pdf_files = self.list_pdf_files(folder_path)
        if pdf_files:
            print("Chatbot: Here are the PDF files:")
            for idx, file in enumerate(pdf_files, start=1):
                print(f"{idx}. {file}")
            try:
                pdf_choice = int(input("Chatbot: Select a PDF by number: "))
                if 1 <= pdf_choice <= len(pdf_files):
                    selected_pdf = pdf_files[pdf_choice - 1]
                    print(f"Chatbot: You selected '{selected_pdf}'.")
                    self.handle_pdf_operations(selected_pdf)
                else:
                    print("Chatbot: Invalid number. Please choose a valid PDF number.")
            except ValueError:
                print("Chatbot: Invalid input! Please enter a valid number.")
        else:
            print("Chatbot: No PDF files found in the folder.")

    def handle_pdf_operations(self, selected_pdf):
        """Perform operations on the selected PDF."""
        print("Chatbot: What do you want to do with this PDF?")
        print("※ 1. Find keywords of ESG report")
        print("※ 2. Topic classifications")
        print("※ 3. All text sentiment analysis")
        print("※ 4. Data mining and/or text analysis methods")
        print("※ 5. Back to the main menu")
        try:
            choice = int(input("Chatbot: Your function choice (1/2/3/4/5): "))
            if choice == 1:
                print("Chatbot: Analysis completed. Close the picture and wait around 10 seconds to continue.")
                analyze_text_file(selected_pdf)
            elif choice == 2:
                print("Chatbot: The topic is", get_topic_by_filename(selected_pdf))
            elif choice == 3:
                print("Chatbot: Sentiment analysis completed. Close the picture to continue.")
                run_sentiment_analysis()
            elif choice == 4:
                function4()
            elif choice == 5:
                print("Chatbot: Returning to the main menu...")
            else:
                print("Chatbot: Invalid choice! Please select 1, 2, 3, 4, 5.")
        except ValueError:
            print("Chatbot: Invalid input! Please enter a valid number.")

    def check_prediction_result(self):
        """Call the model_prediction function to check results."""
        model_prediction()

    def run(self):
        """Run the chatbot and provide the main menu."""
        print("Hi! I am a chatbot for your service")
        while True:
            try:
                print("\nPlease select an option:")
                print("1. Start a daily conversation")
                print("2. List and choose a PDF file")
                print("3. Check the prediction result")
                print("4. Exit")
                choice = int(input("Chatbot: Your choice (1/2/3/4): "))
                if choice == 1:
                    self.start_conversation()
                elif choice == 2:
                    self.handle_pdf_selection()
                elif choice == 3:
                    self.check_prediction_result()
                elif choice == 4:
                    print("Chatbot: Goodbye!")
                    break
                else:
                    print("Invalid choice! Please select 1, 2, 3 or 4.")
            except ValueError:
                print("Invalid input! Please enter a number (1/2/3/4).")

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run()
