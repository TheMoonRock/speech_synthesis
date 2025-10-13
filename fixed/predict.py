import xml.etree.ElementTree as ET
import pandas as pd
import json
import joblib
from catboost import CatBoostClassifier, CatBoostRegressor
import os
import numpy as np

def parse_test_xml(file_path):
    """Парсит тестовый XML без атрибутов времени и ядерных слов"""
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
                
                # В тестовых данных нет nucleus, поэтому ставим False
                has_stress = False
                
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
                    'phrasal_stress': has_stress,  # Будем предсказывать
                    'pause_length': -1,  # Будем предсказывать
                    'is_last_in_intonation': False
                }
                
                current_intonation_words.append(word_data)
                words_in_sentence.append(word_data)
            
            elif elem.tag == 'intonation':
                # Нашли интонационную группу - помечаем последнее слово
                if current_intonation_words:
                    current_intonation_words[-1]['is_last_in_intonation'] = True
                    # Сбрасываем текущую группу
                    current_intonation_words = []
            
            i += 1
        
        # Обрабатываем последнюю группу, если она есть
        if current_intonation_words:
            current_intonation_words[-1]['is_last_in_intonation'] = True
        
        # Обновляем общее количество слов в предложении
        total_words = len(words_in_sentence)
        for word_data in words_in_sentence:
            word_data['total_words_in_sentence'] = total_words
            word_data['words_after'] = total_words - word_data['words_before'] - 1
        
        sentences_data.extend(words_in_sentence)
    
    return pd.DataFrame(sentences_data)

def safe_transform(encoder, data):
    """Безопасное преобразование с обработкой неизвестных значений"""
    try:
        return encoder.transform(data)
    except ValueError:
        # Если встречаются неизвестные значения, заменяем их на наиболее частый класс
        known_data = [x if x in encoder.classes_ else encoder.classes_[0] for x in data]
        return encoder.transform(known_data)

def predict_on_new_data(test_xml_path, output_json_path='submission.json'):
    """Основная функция для предсказания на новых данных"""
    
    print("Загрузка моделей...")
    # Загружаем обученные модели
    stress_model = CatBoostClassifier()
    stress_model.load_model('stress_model.cbm')
    
    pause_model = CatBoostRegressor()
    pause_model.load_model('pause_model.cbm')
    
    # Загружаем кодировщики
    label_encoders = joblib.load('label_encoders.pkl')
    
    print("Парсинг тестовых данных...")
    # Парсим тестовый XML
    test_df = parse_test_xml(test_xml_path)
    print(f"Загружено {len(test_df)} слов из {test_df['sentence_id'].nunique()} предложений")
    
    # Предобработка данных (как в обучении)
    test_df.fillna('', inplace=True)
    
    # Кодируем категориальные переменные с использованием сохраненных кодировщиков
    categorical_columns = ['part_of_speech', 'form', 'gender', 'semantics1', 'semantics2']
    for col in categorical_columns:
        # Используем безопасное преобразование
        test_df[col + '_encoded'] = safe_transform(label_encoders[col], test_df[col].astype(str))
    
    # Признаки для моделей (такие же как при обучении)
    feature_columns = [
        'position_in_sentence',
        'total_words_in_sentence',
        'words_before',
        'words_after',
        'has_capital',
        'word_length',
        'is_last_in_intonation'
    ] + [col + '_encoded' for col in categorical_columns]
    
    print("Выполнение предсказаний...")
    # Предсказываем фразовые ударения
    X_test = test_df[feature_columns]
    stress_predictions = stress_model.predict(X_test)
    
    # Предсказываем длины пауз (только для последних слов в синтагмах)
    pause_predictions = [-1] * len(test_df)  # По умолчанию -1 (нет паузы)
    
    for idx, row in test_df.iterrows():
        if row['is_last_in_intonation']:
            word_features = pd.DataFrame([row[feature_columns]])
            predicted_pause = pause_model.predict(word_features)[0]
            # Округляем и проверяем, что не отрицательное
            pause_predictions[idx] = max(0, int(round(predicted_pause)))
    
    print("Формирование JSON результата...")
    # Создаем JSON результат
    result = []
    
    for sentence_id in sorted(test_df['sentence_id'].unique()):
        sentence_data = test_df[test_df['sentence_id'] == sentence_id]
        
        sentence_result = {"words": []}
        
        for idx, (_, word_row) in enumerate(sentence_data.iterrows()):
            # Получаем соответствующие предсказания
            # Находим индекс в общем массиве предсказаний
            global_idx = sentence_data.index[idx]
            stress_idx = list(test_df.index).index(global_idx)
            pause_idx = stress_idx  # Индексы совпадают
            
            stress_pred = stress_predictions[stress_idx]
            pause_pred = pause_predictions[pause_idx]
            
            sentence_result["words"].append({
                "content": word_row['original'],
                "phrasal_stress": bool(stress_pred),
                "pause_len": pause_pred
            })
        
        result.append(sentence_result)
    
    # Сохраняем результат
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"Результат сохранен в файл: {output_json_path}")
    print(f"Обработано {len(result)} предложений")
    
    # Статистика предсказаний
    stress_count = sum(stress_predictions)
    pause_count = sum(1 for p in pause_predictions if p > 0)
    
    print(f"\nСтатистика предсказаний:")
    print(f"- Ударных слов: {stress_count} ({stress_count/len(stress_predictions)*100:.1f}%)")
    print(f"- Слов с паузами: {pause_count}")
    
    return result

# Использование:
if __name__ == "__main__":
    test_xml_path = "/home/artyom/itmo/syntes_rechi/fixed/Test_Procody.xml"
    
    # Выполняем предсказание
    result = predict_on_new_data(test_xml_path, 'submission.json')
    
    print("\nПредсказание завершено!")