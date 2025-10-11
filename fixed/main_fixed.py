import xml.etree.ElementTree as ET
import pandas as pd
import json
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import joblib
import os

def parse_xml_corrected(file_path):
    """Парсит XML с правильной структурой для вашего файла"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    sentences_data = []
    
    # Ищем все предложения
    for sentence_idx, sentence in enumerate(root.findall('.//sentence')):
        words_in_sentence = []
        current_intonation_words = []
        
        # Собираем все элементы предложения
        elements = list(sentence)
        
        i = 0
        while i < len(elements):
            elem = elements[i]
            
            if elem.tag == 'word':
                original = elem.get('original', '')
                has_stress = elem.get('nucleus') == '2'
                
                # Лингвистические характеристики
                dictitem = elem.find('dictitem')
                if dictitem is not None:
                    pos = dictitem.get('subpart_of_speech', '')
                    form = dictitem.get('form', '')
                    gender = dictitem.get('genesys', '')
                    semantics1 = dictitem.get('semantics1', '')
                    semantics2 = dictitem.get('semantics2', '')
                else:
                    pos = form = gender = semantics1 = semantics2 = ''
                
                word_data = {
                    'sentence_id': sentence_idx,
                    'original': original,
                    'position_in_sentence': len(words_in_sentence),
                    'total_words_in_sentence': 0,  # Заполним позже
                    'words_before': len(words_in_sentence),
                    'words_after': 0,  # Заполним позже
                    'has_capital': original and original[0].isupper(),
                    'word_length': len(original),
                    'part_of_speech': pos,
                    'form': form,
                    'gender': gender,
                    'semantics1': semantics1,
                    'semantics2': semantics2,
                    'phrasal_stress': has_stress,
                    'pause_length': -1,  # По умолчанию нет паузы
                    'is_last_in_intonation': False  # По умолчанию не последнее в синтагме
                }
                
                current_intonation_words.append(word_data)
                words_in_sentence.append(word_data)
            
            elif elem.tag == 'intonation':
                # Нашли интонационную группу - назначаем паузу последнему слову в текущей группе
                if current_intonation_words:
                    # Ищем паузу после интонации
                    if i + 1 < len(elements) and elements[i + 1].tag == 'pause':
                        pause_elem = elements[i + 1]
                        pause_time = pause_elem.get('time')
                        if pause_time and pause_time.isdigit():
                            # Назначаем паузу последнему слову в текущей группе
                            current_intonation_words[-1]['pause_length'] = int(pause_time)
                            current_intonation_words[-1]['is_last_in_intonation'] = True
                            i += 1  # Пропускаем паузу, так как мы её обработали
                    
                    # Сбрасываем текущую группу
                    current_intonation_words = []
            
            i += 1
        
        # Обрабатываем последнюю группу, если она есть
        if current_intonation_words:
            # Для последней группы ищем паузу в конце предложения
            # Ищем последний элемент предложения
            if elements and elements[-1].tag == 'pause':
                pause_elem = elements[-1]
                pause_time = pause_elem.get('time')
                if pause_time and pause_time.isdigit():
                    current_intonation_words[-1]['pause_length'] = int(pause_time)
                    current_intonation_words[-1]['is_last_in_intonation'] = True
        
        # Обновляем общее количество слов в предложении
        total_words = len(words_in_sentence)
        for word_data in words_in_sentence:
            word_data['total_words_in_sentence'] = total_words
            word_data['words_after'] = total_words - word_data['words_before'] - 1
        
        sentences_data.extend(words_in_sentence)
    
    return pd.DataFrame(sentences_data)

# Загружаем данные с исправленной логикой
print("Загрузка и парсинг XML файла...")
df = parse_xml_corrected('/home/artyom/itmo/syntes_rechi/fixed/gogol_utf8_cut.Result.xml')
print(f"Загружено {len(df)} слов")

if len(df) == 0:
    print("Файл не содержит данных. Проверьте структуру XML.")
    exit()

print(f"Уникальных предложений: {df['sentence_id'].nunique()}")

# Статистика
print(f"\nФразовые ударения:")
print(df['phrasal_stress'].value_counts(normalize=True))

pause_data = df[df['pause_length'] > 0]
print(f"\nПаузы (только > 0):")
print(f"Всего пауз: {len(pause_data)}")
if len(pause_data) > 0:
    print(f"Средняя длина: {pause_data['pause_length'].mean():.1f} мс")
    print(f"Мин-Макс: {pause_data['pause_length'].min()}-{pause_data['pause_length'].max()} мс")
    print(f"Примеры слов с паузами:")
    print(pause_data[['original', 'pause_length', 'is_last_in_intonation']].head(10))

# Анализ данных
print("\nРаспределение частей речи:")
print(df['part_of_speech'].value_counts().head(10))

print("\nРаспределение форм слов:")
print(df['form'].value_counts().head(10))

# Предобработка данных
print("\nПредобработка данных...")

# Заполняем пропуски
df.fillna('', inplace=True)

# Кодируем категориальные переменные
categorical_columns = ['part_of_speech', 'form', 'gender', 'semantics1', 'semantics2']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Признаки для моделей
feature_columns = [
    'position_in_sentence',
    'total_words_in_sentence',
    'words_before',
    'words_after',
    'has_capital',
    'word_length',
    'is_last_in_intonation'
] + [col + '_encoded' for col in categorical_columns]

print(f"Используется {len(feature_columns)} признаков:")

# Подготовка данных для двух задач
print("\nПодготовка данных для двух задач...")

# Задача 1: Классификация фразового ударения
X_stress = df[feature_columns]
y_stress = df['phrasal_stress']

# Задача 2: Регрессия для длины пауз (только слова с паузами)
pause_data = df[df['pause_length'] > 0]
X_pause = pause_data[feature_columns]
y_pause = pause_data['pause_length']

print(f"Задача 1 (ударения): {X_stress.shape[0]} примеров")
print(f"Задача 2 (паузы): {X_pause.shape[0]} примеров")

# Разделение на train/test
X_train_stress, X_test_stress, y_train_stress, y_test_stress = train_test_split(
    X_stress, y_stress, test_size=0.2, random_state=42, stratify=y_stress
)

if len(X_pause) > 0:
    X_train_pause, X_test_pause, y_train_pause, y_test_pause = train_test_split(
        X_pause, y_pause, test_size=0.2, random_state=42
    )
else:
    X_train_pause, X_test_pause, y_train_pause, y_test_pause = None, None, None, None

print(f"\nРазделение данных:")
print(f"Ударения - train: {X_train_stress.shape[0]}, test: {X_test_stress.shape[0]}")
if X_train_pause is not None:
    print(f"Паузы - train: {X_train_pause.shape[0]}, test: {X_test_pause.shape[0]}")

# Обучение моделей
print("\nОбучение моделей...")

# Модель для фразового ударения
stress_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=100,
    class_weights=[1, 3] if len(y_stress.unique()) > 1 else None
)

print("Обучение модели для фразового ударения...")
stress_model.fit(
    X_train_stress, y_train_stress,
    eval_set=(X_test_stress, y_test_stress),
    early_stopping_rounds=50
)

# Модель для длины пауз
pause_model = None
if X_train_pause is not None and len(X_train_pause) > 0:
    pause_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_state=42,
        verbose=100
    )

    print("\nОбучение модели для длины пауз...")
    pause_model.fit(
        X_train_pause, y_train_pause,
        eval_set=(X_test_pause, y_test_pause),
        early_stopping_rounds=50
    )

# Оценка моделей
print("\nОценка моделей...")

# Оценка классификации
y_pred_stress = stress_model.predict(X_test_stress)
print("Фразовое ударение - отчет классификации:")
print(classification_report(y_test_stress, y_pred_stress))

# Оценка регрессии
if pause_model is not None and X_test_pause is not None:
    y_pred_pause = pause_model.predict(X_test_pause)
    mse = mean_squared_error(y_test_pause, y_pred_pause)
    mae = mean_absolute_error(y_test_pause, y_pred_pause)

    print(f"\nДлина пауз - метрики регрессии:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f} мс")
    print(f"RMSE: {np.sqrt(mse):.2f} мс")

# Важность признаков
print("\nВажность признаков для фразового ударения:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': stress_model.get_feature_importance()
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))

if pause_model is not None:
    print("\nВажность признаков для длины пауз:")
    feature_importance_pause = pd.DataFrame({
        'feature': feature_columns,
        'importance': pause_model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    print(feature_importance_pause.head(10))

# Функция для предсказания на новых данных
def predict_and_create_json(stress_model, pause_model, label_encoders, df, output_file='result.json'):
    """Создает JSON результат с предсказаниями"""
    
    result = []
    
    # Обрабатываем все предложения
    for sentence_id in df['sentence_id'].unique():
        sentence_data = df[df['sentence_id'] == sentence_id]
        
        sentence_result = {"words": []}
        
        for _, word_row in sentence_data.iterrows():
            # Подготавливаем признаки для предсказания
            word_features = pd.DataFrame([word_row[feature_columns]])
            
            # Предсказываем ударение
            stress_pred = stress_model.predict(word_features)[0]
            
            # Предсказываем длину паузы
            pause_pred = -1
            if pause_model is not None and word_row['is_last_in_intonation']:
                predicted_pause = pause_model.predict(word_features)[0]
                # Округляем и проверяем, что не отрицательное
                pause_pred = max(0, int(round(predicted_pause)))
            
            # Добавляем слово в результат
            sentence_result["words"].append({
                "content": word_row['original'],
                "phrasal_stress": bool(stress_pred),
                "pause_len": pause_pred
            })
        
        result.append(sentence_result)
    
    # Сохраняем результат
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"JSON-результат сохранен в файл: {output_file}")
    print(f"Обработано {len(result)} предложений")
    
    return result

# Генерируем и сохраняем результат
print("\nГенерация JSON-результата...")
json_result = predict_and_create_json(stress_model, pause_model, label_encoders, df, 'lab_submission.json')

# Сохраняем модели
print("\nСохранение моделей...")
stress_model.save_model('stress_model.cbm')
if pause_model is not None:
    pause_model.save_model('pause_model.cbm')

joblib.dump(label_encoders, 'label_encoders.pkl')
print("Модели и кодировщики сохранены")

# Проверяем созданные файлы
print(f"\nСозданные файлы:")
for file in ['lab_submission.json', 'stress_model.cbm', 'label_encoders.pkl']:
    if os.path.exists(file):
        file_size = os.path.getsize(file)
        print(f"✓ {file} ({file_size} bytes)")
    else:
        print(f"✗ {file} - не найден")

if pause_model is not None and os.path.exists('pause_model.cbm'):
    file_size = os.path.getsize('pause_model.cbm')
    print(f"✓ pause_model.cbm ({file_size} bytes)")

print("\nОбработка завершена!")

'''
Загрузка и парсинг XML файла...
Загружено 75956 слов
Уникальных предложений: 5001

Фразовые ударения:
phrasal_stress
False    0.763837
True     0.236163
Name: proportion, dtype: float64

Паузы (только > 0):
Всего пауз: 17910
Средняя длина: 384.1 мс
Мин-Макс: 38-1657 мс
Примеры слов с паузами:
           original  pause_length  is_last_in_intonation
1              души          1020                   True
3            Гоголь          1376                   True
5            ПЕРВЫЙ          1657                   True
7            первая          1020                   True
14          въехала            58                   True
19          бричка,           116                   True
23       холостяки:           174                   True
25   подполковники,           167                   True
26  штабс-капитаны,           208                   True
27        помещики,           249                   True

Распределение частей речи:
part_of_speech
9     20128
1     15593
6     12607
3      8872
10     6897
2      4495
11     3201
7      1309
0      1069
12      843
Name: count, dtype: int64

Распределение форм слов:
form
0     27445
1      7929
2      3631
5      3559
67     3464
60     2073
6      1773
9      1652
20     1484
10     1460
Name: count, dtype: int64

Предобработка данных...
Используется 12 признаков:

Подготовка данных для двух задач...
Задача 1 (ударения): 75956 примеров
Задача 2 (паузы): 17910 примеров

Разделение данных:
Ударения - train: 60764, test: 15192
Паузы - train: 14328, test: 3582

Обучение моделей...
Обучение модели для фразового ударения...
0:      learn: 0.5637750        test: 0.5641038 best: 0.5641038 (0)     total: 57.8ms   remaining: 28.8s
100:    learn: 0.1375239        test: 0.1411039 best: 0.1411039 (100)   total: 382ms    remaining: 1.51s
200:    learn: 0.1275385        test: 0.1380889 best: 0.1380261 (197)   total: 651ms    remaining: 968ms
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.1380260924
bestIteration = 197

Shrink model to first 198 iterations.

Обучение модели для длины пауз...
0:      learn: 234.9492161      test: 234.2065306       best: 234.2065306 (0)   total: 1.48ms   remaining: 738ms
100:    learn: 134.6656524      test: 137.9515672       best: 137.8662257 (68)  total: 93.1ms   remaining: 368ms
Stopped by overfitting detector  (50 iterations wait)

bestTest = 137.8662257
bestIteration = 68

Shrink model to first 69 iterations.

Оценка моделей...
Фразовое ударение - отчет классификации:
              precision    recall  f1-score   support

       False       0.98      0.99      0.98     11604
        True       0.95      0.93      0.94      3588

    accuracy                           0.97     15192
   macro avg       0.96      0.96      0.96     15192
weighted avg       0.97      0.97      0.97     15192


Длина пауз - метрики регрессии:
MSE: 19007.10
MAE: 100.42 мс
RMSE: 137.87 мс

Важность признаков для фразового ударения:
                    feature  importance
6     is_last_in_intonation   68.665652
5               word_length    9.361455
11       semantics2_encoded    4.686051
7    part_of_speech_encoded    3.782649
8              form_encoded    3.277261
3               words_after    2.355693
1   total_words_in_sentence    2.328196
2              words_before    1.631165
0      position_in_sentence    1.520171
9            gender_encoded    1.296357

Важность признаков для длины пауз:
                    feature  importance
3               words_after   95.668621
0      position_in_sentence    0.637401
2              words_before    0.577541
8              form_encoded    0.544024
7    part_of_speech_encoded    0.536925
1   total_words_in_sentence    0.519007
5               word_length    0.433379
10       semantics1_encoded    0.406228
4               has_capital    0.321171
9            gender_encoded    0.214146

Генерация JSON-результата...
JSON-результат сохранен в файл: lab_submission.json
Обработано 5001 предложений

Сохранение моделей...
Модели и кодировщики сохранены

Созданные файлы:
✓ lab_submission.json (11117980 bytes)
✓ stress_model.cbm (230720 bytes)
✓ label_encoders.pkl (1841 bytes)
✓ pause_model.cbm (87464 bytes)

Обработка завершена!

'''