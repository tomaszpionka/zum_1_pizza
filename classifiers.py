from transformers import pipeline
from translate import Translator

reference_dict = {
    'tomato sauce, garlic clove (without mozzarella cheese), Marinara': 'Marinara',
    'tomato sauce, mozzarella, and Parmesan, Margherita': 'Margherita',
    'tomato sauce, mozzarella, cooked ham, pineapple, and Parmesan, Hawaiian': 'Hawaiian',
    'tomato sauce, mozzarella, Parma ham, artichokes, and Parmesan, Parma': 'Parma',
    'tomato sauce, mozzarella, arugula, Parma ham, cherry tomatoes, and Parmesan, Rucola': 'Rucola',
    'tomato sauce, mozzarella, champignons, and Parmesan, Funghi': 'Funghi',
    'tomato sauce, mozzarella, spinata (Italian spicy salami), and jalapeno peppers, Diavolo': 'Diavolo',
    'tomato sauce, mozzarella, cooked ham, mushrooms, black olives, and Parmesan, Capricciosa': 'Capricciosa',
    'tomato sauce, mozzarella, black olives, champignons, and corn, Vegetarian': 'Vegetarian'
}

reference_labels = [
    'tomato sauce, garlic clove (without mozzarella cheese), Marinara',
    'tomato sauce, mozzarella, and Parmesan, Margherita',
    'tomato sauce, mozzarella, cooked ham, pineapple, and Parmesan, Hawaiian',
    'tomato sauce, mozzarella, Parma ham, artichokes, and Parmesan, Parma'
    'tomato sauce, mozzarella, arugula, Parma ham, cherry tomatoes, and Parmesan, Rucola',
    'tomato sauce, mozzarella, champignons, and Parmesan, Funghi',
    'tomato sauce, mozzarella, spinata (Italian spicy salami), and jalapeno peppers, Diavolo',
    'tomato sauce, mozzarella, cooked ham, mushrooms, black olives, and Parmesan, Capricciosa',
    'tomato sauce, mozzarella, black olives, champignons, and corn, Vegetarian'
]


def pizza_classifier(speech_string: str):
    result = None
    translator = Translator(to_lang='en', from_lang='pl')
    translation = str(translator.translate(speech_string))
    classifier = pipeline('zero-shot-classification')
    result = classifier(translation, candidate_labels=reference_labels, )
    pizza_dict = reference_dict
    scores = result['scores']
    labels = result['labels']
    res = dict(zip(labels, scores))
    return pizza_dict[max(res, key=res.get)]


def answer_classifier(speech_string: str):
    translator = Translator(to_lang='en', from_lang='pl')
    translation = str(translator.translate(speech_string))
    classifier = pipeline('zero-shot-classification')
    result = classifier(translation, candidate_labels=['yes', 'no'], )
    answer_dict = {
        'yes': True,
        'no': False
    }
    scores = result['scores']
    labels = result['labels']
    res = dict(zip(labels, scores))
    return answer_dict[max(res, key=res.get)]
