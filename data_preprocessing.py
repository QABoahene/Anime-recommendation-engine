# Data Preprocessing File

## Importing modules
import json
import pandas
import os
from nltk import tokenize
import string
from empath import Empath

## A function to convert topics to empath themes
def topics_to_empath(topic):
    lexicon = Empath()
    categories = [
    'dance', 'money','wedding','domestic_work','sleep',
    'medical_emergency','cold','hate','cheerfulness','aggression','occupation',
    'envy','family','vacation','crime','attractive','masculine',
    'prison','health','pride','dispute','nervousness','government','weakness',
    'horror','swearing_terms','leisure','suffering','royalty','wealthy','tourism',
    'school','magic','beach','journalism','banking','social_media',
    'exercise','kill','art','ridicule','play','computer',
    'college','optimism','stealing','real_estate','home','sexual','fear',
    'irritability','superhero','business','driving','pet','childish','cooking',
    'exasperation','religion','hipster','internet','surprise','reading','worship',
    'leader','independence','body','noise','eating','medieval','zest','confusion',
    'water','sports','death','healing','legend','heroic','celebration','restaurant',
    'violence','dominant_heirarchical','military','neglect','swimming','exotic','love',
    'communication','hearing','order','sympathy','hygiene','weather','anonymity','trust',
    'ancient','deception','fight','dominant_personality','music','vehicle','politeness',
    'farming','meeting','war','urban','shopping','disgust','fire','tool','phone','gain',
    'sound','injury','sailing','rage','science','work','appearance','valuable','warmth',
    'youth','sadness','fun','emotional','joy','affection','traveling','fashion','ugliness',
    'lust','shame','torment','economics','anger','politics','ship','clothing','car','strength',
    'technology','power','animal','party','terrorism','disappointment','poor','plant','pain',
    'beauty','timidity','philosophy','negotiate','negative_emotion','cleaning','messaging',
    'competing','law','friends','achievement','alcohol','feminine','weapon','children',
    'monster','contentment','writing','rural','positive_emotion','musical'
    ]
    return lexicon.analyze(topic, categories = categories)


## A function to change the empath themes to tags
def empath_to_tags(emp):
    lst = sorted([k for k, v in emp.items() if v != 0], key=lambda x:x[1], reverse=True)
    return lst


## A function to strip the themes
def strip_themes(st):
    st = str(st)
    table = str.maketrans(dict.fromkeys(string.punctuation))
    new_s = st.translate(table)
    return new_s
