import pycld2 as cld2
from spacy_langdetect import LanguageDetector
import spacy

class PreProcess:
    def __init__(self) -> None:
        pass


def test():
    test = "Hello everyone"
    _, _, _, detected_language = cld2.detect(test, returnVectors=True)
    print(detected_language)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(test)
    print(doc.text)

    # text_content = "No caption needed for this pic! #सरकार_MSP_केंद्र_खोलो #StopHateAgainstFarmers #FarmersProtest https://t.co/rtUh5mWBu9"
    # text_content = text_content.replace('#', 'AAA')
    # print(text_content)
    # _, _, _, detected_language = cld2.detect(text_content, returnVectors=True)
    # print(detected_language)
    # new_string = ""
    # for (start, end, language, _) in detected_language:
    #     if language == "ENGLISH":
    #         new_string += text_content[start:start+end]

    # print(new_string)


if __name__ == "__main__":
    test()
