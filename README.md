# **Text statistics**
## **Text statistics** - инструмент для статистического анализа текста на русском языке.
Данная библиотека представляет основные подходы к статистическому анализу: частотный анализ, базовые статистики, частотный анализ морфологии и лексическая плотность.

Её функционал может быть использован для следующих задач:
1. Классификация текстов
2. Определение автора
3. Определение доменной области
4. Кластеризация текстов
5. Первичный и визуальный анализ текстовых данных
6. Обнаружение аномалий в текстовых данных


**Работа с пакетом**

*Инициализация*

Для статистического анализа текста создадим объект класса TextStatistics, который принимает на вход текст (строку на русском языке).
stats = TextStatistics(text)

*Базовые статистики*

С помощью метода calculate_main_statics() мы можем получить базовые статистики: среднее число слов в предложении, среднее число букв в слове, число уникальных слов в тексте, число уникальных словоформ в тексте.

stats.calculate_main_statics()

*Частотный анализ*

Метод get_frequent_statistics() расчитывает частоты слов и словарных н-грам в тексте. Он имеет несколько опциональных параметров:
- max_n_gram - максимальная длина словосочетания, для которого считаем статистики. Если =2, то статистики будут расчитаны для отдельных слов и биграм
- preprocess - нормализовать и избавиться от пунктуации перед расчетом. Рекомендуется True
- plot_most_frequent - рисует самые частотные слова на гистограмме
- n_freq - если plot_most_frequent указано, то сколько самых частотных нарисовать на графике.

stats.get_frequent_statistics(plot_most_frequent=False)

*Морфологический частотный анализ*

Метод get_morph_statistics() поможет понять, какой морфологией чаще всего пользуется автор. Метод возвращает словарь объектов Counter(). В словарь включены следующие характеристики:
- parts_of_speech - часть речи
- animacy - одушевленность
- aspect - вид
- case - падеж
- gender - род
- number - число
- tense - время

Данный метод является надстройкой над анализатором pymorphy2, с обозначениями можно ознакомиться в [документации pymorphy2](https://pymorphy2.readthedocs.io/en/0.2/user/index.html)

stats.get_morph_statistics()

*Лексическая плотность*

Данный метод позволяет определить, с какой вероятностью текст принадлежит той или иной предметной области, на основе словаря терминов этой предметной области.
Вычисляется лексическая плотность - число слов-терминов, деленное на общее число слов текста.
Принимает на вход term_list - список слов-терминов, характеризующих предметную область.

stats.get_lexical_density(terms)

**Пример использования основного функционала вы можете найти в ноутбуке Usage example.ipynb**

*Автор - Малышева Анастасия/Malysheva Anastasia*

*For Application aspects of Social Data Processing by Anton Kolonin course work.*
