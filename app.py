import pandas as pd
import os

# os.environ["PATH"] += ":/Users/tomaszpionka/audio-orchestrator-ffmpeg/bin"

price_mapping = {
    'Marinara': 23,
    'Margherita': 25,
    'Hawaiian': 34,
    'Parma': 37,
    'Rucola': 35,
    'Funghi': 32,
    'Diavolo': 37,
    'Capricciosa': 35,
    'Vegetarian': 30
}

pizza_names = [
    'marinara', 'margherita', 'hawaiian',
    'parma', 'rucola', 'funghi',
    'diavolo', 'capricciosa', 'vegetarian'
]

prices = [
    23, 25, 34,
    37, 35, 32,
    37, 35, 30
]

ingredients = [
    'sos pomidorowy, ząbek czosnku, (bez sera mozarella)', 'sos pomidorowy, mozarella, parmezan',
    'sos pomidorowy, mozarella, szynka gotowana, ananas, parmezan',
    'sos pomidorowy, mozarella, szynka parmeńska, karczochy, parmezan',
    'sos pomidorowy, mozarella, rukola, szynka parmeńska, pomidorki cherry, parmezan',
    'sos pomidorowy, mozarella, pieczarki, parmezan',
    'sos pomidorowy, mozarella, spinata (włoskie salami pikantne), papryczki jalapeno',
    'sos pomidorowy, mozarella, szynka gotowana, pieczarki, czarne oliwki, parmezan',
    'sos pomidorowy, mozarella, czarne oliwki, pieczarki, kukurydza'
]

pizza_menu = pd.DataFrame(
    {
        'Pizza': pizza_names,
        'Ingredients': ingredients,
        'Price': prices
    }
)


def order():
    import classifiers as c
    import audio_util as au
    filepath = str(os.getcwd()) + "/recordings/input.wav"
    filepath_output = str(os.getcwd()) + "/recordings/output.wav"

    print(pizza_menu.head(10))

    au.str_to_speech(f"Dzień dobry, proszę podaj pizzę jaką chcesz zamówić!")
    pizza_choice = au.speech_to_str(filepath, filepath_output)
    pizza_classified = c.pizza_classifier(pizza_choice)
    au.str_to_speech(f"Wybrano pizzę: {pizza_classified}.")

    au.str_to_speech(f"Podaj adres, pod który dostarczymy zamówienie.")
    address_choice = au.speech_to_str(filepath, filepath_output)
    au.str_to_speech(f"Twój adres: {address_choice}")

    au.str_to_speech(f"Czy potwierdzasz powyższe informacje?")
    confirmation = au.speech_to_str(filepath, filepath_output)
    confirmation_classified = c.answer_classifier(confirmation)
    if confirmation_classified:
        au.str_to_speech(f"Potwierdzam zamówienie. Pizza {pizza_classified}, na adres {address_choice}")
    else:
        au.str_to_speech(f"Spróbuj ponownie!")
        order()


if __name__ == "__main__":
    order()
