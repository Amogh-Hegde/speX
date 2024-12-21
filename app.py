from speech_recognition import listen_and_respond
from face_recognition_util import identify_person
from text_reader import read_text
from datetime import datetime, date
from pyttsx3 import init
from num2words import num2words

# Text-to-speech engine initialization
engine = init()
engine.setProperty('rate', 150)

def respond_to_command(command):
    if 'time' in command:
        now = datetime.now()
        hour = int(now.strftime("%I"))
        minute = int(now.strftime("%M"))
        response = f"It is {num2words(hour)} {num2words(minute)} {now.strftime('%p')} now."
        engine.say(response)
        engine.runAndWait()

    elif 'date' in command or "today's" in command:
        today = date.today()
        engine.say(f"Today's date is {today.strftime('%B %d, %Y')}")
        engine.runAndWait()

    elif 'read' in command or 'text' in command:
        read_text(engine)

    elif 'who' in command or 'person' in command:
        identify_person(engine)

if __name__ == "__main__":
    while True:
        command = listen_and_respond()
        if command:
            respond_to_command(command)
