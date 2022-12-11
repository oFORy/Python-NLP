import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Загружаем дата сет
news_d = pd.read_csv("train.csv")
submit_test = pd.read_csv("test.csv")

# Выводим формы и столбцы тренировочного датасета
print(" Форма данных новостей: ", news_d.shape)
print(" Столбцы данных новостей:", news_d.columns)

## используя df.head(), мы можем сразу ознакомиться с набором данных.
news_d.head()

#Стартистика Text Word: минимальное среднее, максимальное и межквартильный диапазон

txt_length = news_d.text.str.split().str.len()
txt_length.describe()

#Статистика титулов

title_length = news_d.title.str.split().str.len()
title_length.describe()

sns.countplot(x="label", data=news_d)
print("1: Ненадежный")
print("0: Надежный")
print("Распределение столбцов:")
print(news_d.label.value_counts())

print(round(news_d.label.value_counts(normalize=True),2)*100)

# Константы, которые используются для очистки наборов данных
column_n = ['id', 'title', 'author', 'text', 'label']
remove_c = ['id','author']
categorical_features = []
target_col = ['label']
text_f = ['title', 'text']

# Чистим наборы данных
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from collections import Counter

ps = PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()

stop_words = stopwords.words('russian') # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
stopwords_dict = Counter(stop_words)

# Удалить неиспользуемые столбцы
def remove_unused_c(df,column_n=remove_c):
    df = df.drop(column_n,axis=1)
    return df

# Вставляем нулевые значения с None
def null_process(feature_df):
    for col in text_f:
        feature_df.loc[feature_df[col].isnull(), col] = "None"
    return feature_df

def clean_dataset(df):
    # удалить неиспользуемый столбец
    df = remove_unused_c(df)
    # вставить нулевое значение
    df = null_process(df)
    return df

# Очистка текста от неиспользуемых символов
def clean_text(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # removing urls
    text = str(text).replace(r'[^\.\w\s]', ' ')  # remove everything but characters and punctuation
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    #text = ' '.join(text)
    return text

## Предварительная обработка Nltk включает:
# Стоп-слова, стемминг и лемматизация
# Для проекта использую только удаление стоп-слов
def nltk_preprocess(text):
    text = clean_text(text)
    wordlist = re.sub(r'[^\w\s]', '', text).split()
    text = ' '.join([wnl.lemmatize(word) for word in wordlist if word not in stopwords_dict])
    return  text

# Выполните очистку данных в наборе обучающих и тестовых данных, вызвав функцию clean_dataset.
df = clean_dataset(news_d)
# применить предварительную обработку текста с помощью метода применения, вызвав функцию nltk_preprocess
df["text"] = df.text.apply(nltk_preprocess)
# применить предварительную обработку к заголовку с помощью метода применения, вызвав функцию nltk_preprocess
df["title"] = df.title.apply(nltk_preprocess)

# Набор данных после этапа очистки и предварительной обработки
df.head()

from wordcloud import WordCloud, STOPWORDS

# инициализировать облако слов
wordcloud = WordCloud( background_color='black', width=800, height=600)
# сгенерировать облако слов, передав корпус
text_cloud = wordcloud.generate(' '.join(df['text']))
# построение облака слов
plt.figure(figsize=(20,30))
plt.imshow(text_cloud)
plt.axis('off')
plt.show()

true_n = ' '.join(df[df['label']==0]['text'])
wc = wordcloud.generate(true_n)
plt.figure(figsize=(20,30))
plt.imshow(wc)
plt.axis('off')
plt.show()

fake_n = ' '.join(df[df['label']==1]['text'])
wc= wordcloud.generate(fake_n)
plt.figure(figsize=(20,30))
plt.imshow(wc)
plt.axis('off')
plt.show()

def plot_top_ngrams(corpus, title, ylabel, xlabel="Количество вхождений", n=2):
  true_b = (pd.Series(nltk.ngrams(corpus.split(), n)).value_counts())[:20]
  true_b.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
  plt.title(title)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.show()

plot_top_ngrams(true_n, 'Топ-20 наиболее часто встречающихся биграмм правдивых новостей', "Биграммы", n=2)

plot_top_ngrams(fake_n, 'Топ-20 наиболее часто встречающихся фейковых новостей биграмм', "Биграммы", n=2)

plot_top_ngrams(true_n, 'Топ-20 наиболее часто встречающихся триграмм правдивых новостей', "Триграммы", n=3)

plot_top_ngrams(fake_n, 'Топ-20 наиболее часто встречающихся триграмм фейковых новостей', "Триграммы", n=3)

import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ безопасно вызывать эту функцию, даже если cuda недоступна
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

set_seed(1)

# модель, которую мы будем тренировать, базовый BERT без корпуса
# проверьте модели классификации текста здесь: https://huggingface.co/models?filter=text-classification
model_name = "bert-base-uncased"
# максимальная длина последовательности для каждого образца документа/предложения
max_length = 512

# загрузить токенизатор
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

news_df = news_d[news_d['text'].notna()]
news_df = news_df[news_df["author"].notna()]
news_df = news_df[news_df["title"].notna()]

def prepare_data(df, test_size=0.2, include_title=True, include_author=True):
  texts = []
  labels = []
  for i in range(len(df)):
    text = df["text"].iloc[i]
    label = df["label"].iloc[i]
    if include_title:
      text = df["title"].iloc[i] + " - " + text
    if include_author:
      text = df["author"].iloc[i] + " : " + text
    if text and label in [0, 1]:
      texts.append(text)
      labels.append(label)
  return train_test_split(texts, labels, test_size=test_size)

train_texts, valid_texts, train_labels, valid_labels = prepare_data(news_df)

print(len(train_texts), len(train_labels))
print(len(valid_texts), len(valid_labels))

# токенизировать набор данных, обрезать при передаче `max_length`,
# и дополнить 0, если меньше `max_length`
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# преобразовать наши токенизированные данные в набор данных факела
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

# загрузка модели
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

from sklearn.metrics import accuracy_score

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # рассчитать точность, используя функцию sklearn
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

training_args = TrainingArguments(
    output_dir='./results',          # выходной каталог
    num_train_epochs=1,              # общее количество тренировочных эпох
    per_device_train_batch_size=10,  # размер партии на устройство во время обучения
    per_device_eval_batch_size=20,   # размер партии для оценки
    warmup_steps=100,                # количество шагов прогрева для планировщика скорости обучения
    logging_dir='./logs',            # директория для хранения логов
    load_best_model_at_end=True,     # загрузить лучшую модель после завершения обучения (метрика по умолчанию — потеря)
    logging_steps=200,               # регистрировать и сохранять веса каждый logging_steps
    save_steps=200,
    evaluation_strategy="steps",     # оценить каждый `logging_steps`
)

trainer = Trainer(
    model=model,                         # созданная модель Трансформеров для обучения
    args=training_args,                  # обучающие аргументы, определенные выше
    train_dataset=train_dataset,         # обучающий набор данных
    eval_dataset=valid_dataset,          # набор данных для оценки
    compute_metrics=compute_metrics,     # обратный вызов, который вычисляет интересующие метрики
)



#torch.cuda.memory_summary(device=None, abbreviated=False)

# обучить модель
trainer.train()

# оценить текущую модель после обучения
trainer.evaluate()

# сохранение точно настроенной модели и токенизатора
model_path = "fake-news-bert-base-uncased"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

def get_prediction(text, convert_to_label=False):
    # подготовить текст в токенизированную последовательность
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        #.to("cuda")
    # выполнить вывод модели
    outputs = model(**inputs)
    # получить выходные вероятности, выполнив softmax
    probs = outputs[0].softmax(1)
    # выполнение функции argmax для получения метки-кандидата
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
      return d[int(probs.argmax())]
    else:
      return int(probs.argmax())

real_news = """
Tim Tebow Will Attempt Another Comeback, This Time in Baseball - The New York Times",Daniel Victor,"If at first you don’t succeed, try a different sport. Tim Tebow, who was a Heisman   quarterback at the University of Florida but was unable to hold an N. F. L. job, is pursuing a career in Major League Baseball. He will hold a workout for M. L. B. teams this month, his agents told ESPN and other news outlets. “This may sound like a publicity stunt, but nothing could be further from the truth,” said Brodie Van Wagenen,   of CAA Baseball, part of the sports agency CAA Sports, in the statement. “I have seen Tim’s workouts, and people inside and outside the industry - scouts, executives, players and fans  —   will be impressed by his talent. ” It’s been over a decade since Tebow, 28, has played baseball full time, which means a comeback would be no easy task. But the former major league catcher Chad Moeller, who said in the statement that he had been training Tebow in Arizona, said he was “beyond impressed with Tim’s athleticism and swing. ” “I see bat speed and power and real baseball talent,” Moeller said. “I truly believe Tim has the skill set and potential to achieve his goal of playing in the major leagues and based on what I have seen over the past two months, it could happen relatively quickly. ” Or, take it from Gary Sheffield, the former   outfielder. News of Tebow’s attempted comeback in baseball was greeted with skepticism on Twitter. As a junior at Nease High in Ponte Vedra, Fla. Tebow drew the attention of major league scouts, batting . 494 with four home runs as a left fielder. But he ditched the bat and glove in favor of pigskin, leading Florida to two national championships, in 2007 and 2009. Two former scouts for the Los Angeles Angels told WEEI, a Boston radio station, that Tebow had been under consideration as a high school junior. “We wanted to draft him, but he never sent back his information card,” said one of the scouts, Tom Kotchman, referring to a questionnaire the team had sent him. “He had a strong arm and had a lot of power,” said the other scout, Stephen Hargett. “If he would have been there his senior year he definitely would have had a good chance to be drafted. ” “It was just easy for him,” Hargett added. “You thought, If this guy dedicated everything to baseball like he did to football how good could he be?” Tebow’s high school baseball coach, Greg Mullins, told The Sporting News in 2013 that he believed Tebow could have made the major leagues. “He was the leader of the team with his passion, his fire and his energy,” Mullins said. “He loved to play baseball, too. He just had a bigger fire for football. ” Tebow wouldn’t be the first athlete to switch from the N. F. L. to M. L. B. Bo Jackson had one   season as a Kansas City Royal, and Deion Sanders played several years for the Atlanta Braves with mixed success. Though Michael Jordan tried to cross over to baseball from basketball as a in 1994, he did not fare as well playing one year for a Chicago White Sox minor league team. As a football player, Tebow was unable to match his college success in the pros. The Denver Broncos drafted him in the first round of the 2010 N. F. L. Draft, and he quickly developed a reputation for clutch performances, including a memorable   pass against the Pittsburgh Steelers in the 2011 Wild Card round. But his stats and his passing form weren’t pretty, and he spent just two years in Denver before moving to the Jets in 2012, where he spent his last season on an N. F. L. roster. He was cut during preseason from the New England Patriots in 2013 and from the Philadelphia Eagles in 2015.
"""

get_prediction(real_news, convert_to_label=True)

# прочитать набор тестов
test_df = pd.read_csv("test.csv")

test_df.head()

# сделать копию тестового набора
new_df = test_df.copy()

# добавить новый столбец, содержащий автора, заголовок и содержание статьи
new_df["new_text"] = new_df["author"].astype(str) + " : " + new_df["title"].astype(str) + " - " + new_df["text"].astype(str)
new_df.head()

# получить прогноз всего набора тестов
new_df["label"] = new_df["new_text"].apply(get_prediction)

# сделать файл отправки
final_df = new_df[["id", "label"]]
final_df.to_csv("submit_final.csv", index=False)






