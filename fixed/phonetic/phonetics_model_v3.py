import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import re
from collections import Counter
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

# Создаем директорию для результатов
RESULTS_DIR = "/home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files"
os.makedirs(RESULTS_DIR, exist_ok=True)

class PhoneticsDataset(Dataset):
    def __init__(self, words, grapheme_to_idx, allophone_to_idx, max_word_len=50):
        self.words = words
        self.grapheme_to_idx = grapheme_to_idx
        self.allophone_to_idx = allophone_to_idx
        self.max_word_len = max_word_len
        self.idx_to_allophone = {v: k for k, v in allophone_to_idx.items()}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word_data = self.words[idx]
        graphemes = word_data['graphemes']
        allophones = word_data['allophones']
        
        # Convert to indices
        grapheme_indices = [self.grapheme_to_idx.get(g, self.grapheme_to_idx['<UNK>']) for g in graphemes]
        allophone_indices = [self.allophone_to_idx.get(a, self.allophone_to_idx['<UNK>']) for a in allophones]
        
        # Pad sequences
        grapheme_padded = grapheme_indices + [self.grapheme_to_idx['<PAD>']] * (self.max_word_len - len(graphemes))
        allophone_padded = allophone_indices + [self.allophone_to_idx['<PAD>']] * (self.max_word_len - len(allophones))
        
        # Create masks
        grapheme_mask = [1] * len(graphemes) + [0] * (self.max_word_len - len(graphemes))
        allophone_mask = [1] * len(allophones) + [0] * (self.max_word_len - len(allophones))
        
        return (
            torch.tensor(grapheme_padded, dtype=torch.long),
            torch.tensor(allophone_padded, dtype=torch.long),
            torch.tensor(grapheme_mask, dtype=torch.float),
            torch.tensor(allophone_mask, dtype=torch.float)
        )

class SimpleGraphemeToAllophoneModel(nn.Module):
    def __init__(self, grapheme_vocab_size, allophone_vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(SimpleGraphemeToAllophoneModel, self).__init__()
        
        self.embedding = nn.Embedding(grapheme_vocab_size, embedding_dim, padding_idx=0)
        
        # Encoder - bidirectional LSTM
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              batch_first=True, bidirectional=True, dropout=dropout)
        
        # Output layer - directly from encoder outputs
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, allophone_vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, graphemes, grapheme_mask):
        # Embedding
        embedded = self.embedding(graphemes)
        embedded = self.dropout(embedded)
        
        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        
        # Output projection directly from encoder outputs
        output = self.output_layer(encoder_outputs)
        
        return output

def parse_xml_for_phonetics(file_path):
    """Парсит XML для извлечения фонетических данных - исправленная версия"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    words_data = []
    
    print("Парсинг XML файла...")
    for sentence in tqdm(root.findall('.//sentence'), desc="Обработка предложений"):
        for word in sentence.findall('.//word'):
            original = word.get('original', '')
            
            # Извлекаем графемы (буквы) из тегов <letter>
            graphemes = []
            letters = word.findall('.//letter')
            
            for letter in letters:
                char = letter.get('char')
                if char:
                    # Декодируем HTML entities если нужно
                    if '&#' in char:
                        try:
                            # Пытаемся извлечь числовой код
                            code = char[2:-1]  # убираем '&#' и ';'
                            if code.isdigit():
                                char = chr(int(code))
                        except:
                            pass
                    graphemes.append(char.lower())
            
            # Если не нашли букв в тегах, разбиваем original
            if not graphemes and original:
                graphemes = [char.lower() for char in original if char.isalpha()]
            
            # Извлекаем аллофоны
            allophones = []
            for allophone in word.findall('.//allophone'):
                ph = allophone.get('ph')
                if ph:
                    allophones.append(ph)
            
            if graphemes and allophones and len(graphemes) > 0 and len(allophones) > 0:
                words_data.append({
                    'original': original,
                    'graphemes': graphemes,
                    'allophones': allophones
                })
    
    print(f"Пример данных: {words_data[:3] if words_data else 'Нет данных'}")
    return words_data

def parse_full_xml_for_training(file_path):
    """Парсит весь XML для обучения и создания полного JSON"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    all_sentences_data = []
    words_data = []
    
    print("Полный парсинг XML файла...")
    for sentence_idx, sentence in enumerate(tqdm(root.findall('.//sentence'), desc="Обработка предложений")):
        sentence_data = {
            'sentence_id': sentence_idx,
            'words': []
        }
        
        for word in sentence.findall('.//word'):
            original = word.get('original', '')
            
            # Извлекаем графемы (буквы) из тегов <letter>
            graphemes = []
            letters = word.findall('.//letter')
            
            for letter in letters:
                char = letter.get('char')
                if char:
                    # Декодируем HTML entities если нужно
                    if '&#' in char:
                        try:
                            code = char[2:-1]
                            if code.isdigit():
                                char = chr(int(code))
                        except:
                            pass
                    graphemes.append(char.lower())
            
            # Если не нашли букв в тегах, разбиваем original
            if not graphemes and original:
                graphemes = [char.lower() for char in original if char.isalpha()]
            
            # Извлекаем аллофоны
            allophones = []
            for allophone in word.findall('.//allophone'):
                ph = allophone.get('ph')
                if ph:
                    allophones.append(ph)
            
            if graphemes and allophones and len(graphemes) > 0 and len(allophones) > 0:
                word_data = {
                    'original': original,
                    'graphemes': graphemes,
                    'allophones': allophones
                }
                
                words_data.append(word_data)
                sentence_data['words'].append(word_data)
        
        if sentence_data['words']:
            all_sentences_data.append(sentence_data)
    
    print(f"Загружено {len(all_sentences_data)} предложений")
    print(f"Всего слов для обучения: {len(words_data)}")
    return words_data, all_sentences_data

def build_vocabularies(words_data):
    """Строит словари графем и аллофонов"""
    all_graphemes = []
    all_allophones = []
    
    print("Построение словарей...")
    for word in tqdm(words_data, desc="Обработка слов"):
        all_graphemes.extend(word['graphemes'])
        all_allophones.extend(word['allophones'])
    
    print(f"Всего графем: {len(all_graphemes)}, уникальных: {len(set(all_graphemes))}")
    print(f"Всего аллофонов: {len(all_allophones)}, уникальных: {len(set(all_allophones))}")
    
    # Создаем словари
    grapheme_counter = Counter(all_graphemes)
    allophone_counter = Counter(all_allophones)
    
    print(f"Топ-10 графем: {grapheme_counter.most_common(10)}")
    print(f"Топ-10 аллофонов: {allophone_counter.most_common(10)}")
    
    # Специальные токены
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    
    grapheme_to_idx = {token: i for i, token in enumerate(special_tokens)}
    allophone_to_idx = {token: i for i, token in enumerate(special_tokens)}
    
    # Добавляем графемы
    for grapheme, count in tqdm(grapheme_counter.items(), desc="Создание словаря графем"):
        if grapheme not in grapheme_to_idx:
            grapheme_to_idx[grapheme] = len(grapheme_to_idx)
    
    # Добавляем аллофоны
    for allophone, count in tqdm(allophone_counter.items(), desc="Создание словаря аллофонов"):
        if allophone not in allophone_to_idx:
            allophone_to_idx[allophone] = len(allophone_to_idx)
    
    return grapheme_to_idx, allophone_to_idx

def train_phonetics_model():
    print("Загрузка данных для фонетического транскриптора...")
    
    # Парсим XML для обучения
    words_data, all_sentences_data = parse_full_xml_for_training('/home/artyom/itmo/syntes_rechi/fixed/gogol_utf8_cut.Result.xml')
    
    if len(words_data) == 0:
        print("Не найдено данных с фонетической разметкой")
        return None, None, None, None
    
    # Строим словари
    grapheme_to_idx, allophone_to_idx = build_vocabularies(words_data)
    print(f"Размер словаря графем: {len(grapheme_to_idx)}")
    print(f"Размер словаря аллофонов: {len(allophone_to_idx)}")
    
    # Разделяем данные на train/test
    train_data, test_data = train_test_split(words_data, test_size=0.2, random_state=42)
    
    # Создаем датасеты
    train_dataset = PhoneticsDataset(train_data, grapheme_to_idx, allophone_to_idx)
    test_dataset = PhoneticsDataset(test_data, grapheme_to_idx, allophone_to_idx)
    
    # Создаем DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Инициализируем модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    model = SimpleGraphemeToAllophoneModel(
        grapheme_vocab_size=len(grapheme_to_idx),
        allophone_vocab_size=len(allophone_to_idx),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    # Проверяем количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров модели: {total_params:,}")
    
    # Оптимизатор и функция потерь
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=allophone_to_idx['<PAD>'])
    
    # Обучение
    num_epochs = 30
    best_accuracy = 0
    
    print("Начало обучения...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Прогресс-бар для обучения
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (graphemes, allophones, grapheme_mask, allophone_mask) in enumerate(train_pbar):
            graphemes = graphemes.to(device)
            allophones = allophones.to(device)
            grapheme_mask = grapheme_mask.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(graphemes, grapheme_mask)
            
            # Reshape для вычисления потерь
            outputs = outputs.view(-1, outputs.size(-1))
            targets = allophones.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Вычисляем accuracy
            _, predicted = torch.max(outputs, 1)
            mask = targets != allophone_to_idx['<PAD>']
            correct_predictions += ((predicted == targets) & mask).sum().item()
            total_predictions += mask.sum().item()
            
            # Обновляем прогресс-бар
            current_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Валидация
        model.eval()
        val_accuracy = 0
        val_predictions = 0
        val_total = 0
        
        val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for graphemes, allophones, grapheme_mask, allophone_mask in val_pbar:
                graphemes = graphemes.to(device)
                allophones = allophones.to(device)
                grapheme_mask = grapheme_mask.to(device)
                
                outputs = model(graphemes, grapheme_mask)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = allophones.view(-1)
                
                _, predicted = torch.max(outputs, 1)
                mask = targets != allophone_to_idx['<PAD>']
                val_predictions += ((predicted == targets) & mask).sum().item()
                val_total += mask.sum().item()
                
                current_val_acc = val_predictions / val_total if val_total > 0 else 0
                val_pbar.set_postfix({
                    'Val Acc': f'{current_val_acc:.4f}'
                })
        
        val_accuracy = val_predictions / val_total if val_total > 0 else 0
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Сохраняем лучшую модель
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_path = os.path.join(RESULTS_DIR, 'best_phonetics_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'grapheme_to_idx': grapheme_to_idx,
                'allophone_to_idx': allophone_to_idx,
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy
            }, model_path)
            print(f"Новая лучшая модель сохранена с точностью: {best_accuracy:.4f}")
        
        # Ранняя остановка если accuracy хорошая
        if val_accuracy > 0.95:
            print("Достигнута высокая точность, останавливаем обучение")
            break
    
    print("Обучение завершено!")
    return model, grapheme_to_idx, allophone_to_idx, all_sentences_data

def predict_allophones(model, grapheme_to_idx, allophone_to_idx, words):
    """Предсказывает аллофоны для списка слов"""
    device = next(model.parameters()).device
    model.eval()
    
    idx_to_allophone = {v: k for k, v in allophone_to_idx.items()}
    
    results = []
    
    with torch.no_grad():
        for word in words:  # Убрал tqdm отсюда
            # Преобразуем слово в графемы
            graphemes = [char.lower() for char in word if char.isalpha()]
            if not graphemes:
                results.append([])
                continue
            
            # Конвертируем в индексы
            grapheme_indices = [grapheme_to_idx.get(g, grapheme_to_idx['<UNK>']) for g in graphemes]
            grapheme_tensor = torch.tensor([grapheme_indices], dtype=torch.long).to(device)
            grapheme_mask = torch.ones((1, len(graphemes)), dtype=torch.float).to(device)
            
            # Предсказание
            output = model(grapheme_tensor, grapheme_mask)
            _, predicted = torch.max(output, 2)
            
            # Конвертируем обратно в аллофоны
            predicted_allophones = []
            for idx in predicted[0].cpu().numpy():
                allophone = idx_to_allophone.get(idx, '<UNK>')
                if allophone not in ['<PAD>', '<SOS>', '<EOS>']:
                    predicted_allophones.append(allophone)
            
            results.append(predicted_allophones)
    
    return results

def create_complete_phonetics_json(model, grapheme_to_idx, allophone_to_idx, all_sentences_data, output_file='complete_phonetics_result.json'):
    """Создает полный JSON с фонетической транскрипцией для всего XML файла"""
    
    print(f"Создание полной фонетической транскрипции для {len(all_sentences_data)} предложений...")
    
    result = []
    
    for sentence_data in tqdm(all_sentences_data, desc="Обработка предложений"):
        sentence_result = {
            "sentence_id": sentence_data['sentence_id'],
            "words": []
        }
        
        # Собираем все оригинальные слова из предложения
        original_words = [word['original'] for word in sentence_data['words']]
        
        # Предсказываем аллофоны для всех слов предложения
        predicted_allophones = predict_allophones(model, grapheme_to_idx, allophone_to_idx, original_words)
        
        # Формируем результат в формате требуемом для лабораторной работы
        for word_data, allophones in zip(sentence_data['words'], predicted_allophones):
            word_result = {
                "content": word_data['original'],
                "allophones": allophones
            }
            sentence_result["words"].append(word_result)
        
        result.append(sentence_result)
    
    # Сохраняем в файл
    output_path = os.path.join(RESULTS_DIR, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"Полная фонетическая транскрипция сохранена в {output_path}")
    print(f"Обработано {len(result)} предложений, {sum(len(s['words']) for s in result)} слов")
    
    return result

def create_lab_submission_json(model, grapheme_to_idx, allophone_to_idx, all_sentences_data, output_file='lab_phonetics_submission.json'):
    """Создает JSON в формате для сдачи лабораторной работы"""
    
    print(f"Создание JSON для сдачи лабораторной работы...")
    
    result = []
    
    for sentence_data in tqdm(all_sentences_data, desc="Формирование JSON для сдачи"):
        sentence_result = {
            "sentence": "",
            "words": []
        }
        
        # Собираем текст предложения
        original_words = [word['original'] for word in sentence_data['words']]
        sentence_text = " ".join(original_words)
        sentence_result["sentence"] = sentence_text
        
        # Предсказываем аллофоны
        predicted_allophones = predict_allophones(model, grapheme_to_idx, allophone_to_idx, original_words)
        
        # Формируем результат в формате с allophone тегами
        for word_data, allophones in zip(sentence_data['words'], predicted_allophones):
            word_result = {
                "original": word_data['original'],
                "allophones": [{"ph": ph} for ph in allophones]
            }
            sentence_result["words"].append(word_result)
        
        result.append(sentence_result)
    
    # Сохраняем в файл
    output_path = os.path.join(RESULTS_DIR, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"Файл для сдачи лабораторной работы сохранен в {output_path}")
    
    return result

# Функция для тестирования модели
def test_model_predictions(model, grapheme_to_idx, allophone_to_idx):
    """Тестирует модель на нескольких примерах"""
    test_words = ["привет", "мир", "тест", "аллофон", "графема"]
    
    print("\nТестирование модели на примерах:")
    predictions = predict_allophones(model, grapheme_to_idx, allophone_to_idx, test_words)
    
    for word, allophones in zip(test_words, predictions):
        print(f"{word}: {allophones}")

# Основная функция
if __name__ == "__main__":
    print("=" * 60)
    print("Фонетический транскриптор - обучение модели")
    print("=" * 60)
    print(f"Результаты будут сохранены в: {RESULTS_DIR}")
    
    # Обучаем модель и получаем все данные
    model, grapheme_to_idx, allophone_to_idx, all_sentences_data = train_phonetics_model()
    
    if model is not None:
        print("\n" + "=" * 60)
        print("Тестирование обученной модели")
        print("=" * 60)
        
        # Тестируем модель
        test_model_predictions(model, grapheme_to_idx, allophone_to_idx)
        
        print("\n" + "=" * 60)
        print("Создание полной фонетической транскрипции")
        print("=" * 60)
        
        # Создаем полный JSON для всего XML файла
        complete_result = create_complete_phonetics_json(
            model, grapheme_to_idx, allophone_to_idx, all_sentences_data,
            'complete_phonetics_result.json'
        )
        
        # Создаем JSON для сдачи лабораторной работы
        lab_submission = create_lab_submission_json(
            model, grapheme_to_idx, allophone_to_idx, all_sentences_data,
            'lab_phonetics_submission.json'
        )
        
        print(f"\nГотово! Созданы файлы в директории {RESULTS_DIR}:")
        print(f"- best_phonetics_model.pth (обученная модель)")
        print(f"- complete_phonetics_result.json (полная транскрипция всего XML)")
        print(f"- lab_phonetics_submission.json (файл для сдачи лабораторной работы)")
        
        # Показываем статистику
        total_sentences = len(complete_result)
        total_words = sum(len(sentence['words']) for sentence in complete_result)
        total_allophones = sum(len(word['allophones']) for sentence in complete_result for word in sentence['words'])
        
        print(f"\nСтатистика:")
        print(f"- Предложений: {total_sentences}")
        print(f"- Слов: {total_words}")
        print(f"- Аллофонов: {total_allophones}")
        
        # Показываем полные пути к файлам
        print(f"\nПолные пути к файлам:")
        print(f"- Модель: {os.path.join(RESULTS_DIR, 'best_phonetics_model.pth')}")
        print(f"- Полная транскрипция: {os.path.join(RESULTS_DIR, 'complete_phonetics_result.json')}")
        print(f"- Для сдачи: {os.path.join(RESULTS_DIR, 'lab_phonetics_submission.json')}")
        
    else:
        print("Ошибка: модель не была обучена")




'''
artyom@artyom-pc:~/itmo$ source /home/artyom/itmo/itmo_venv/bin/activate
(itmo_venv) artyom@artyom-pc:~/itmo$ /home/artyom/itmo/itmo_venv/bin/python /home/artyom/itmo/syntes_rechi/fixed/phonetic/phonetics_model_v3.py
============================================================
Фонетический транскриптор - обучение модели
============================================================
Результаты будут сохранены в: /home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files
Загрузка данных для фонетического транскриптора...
Полный парсинг XML файла...
Обработка предложений: 100%|█████████████████████████████████████████████| 5001/5001 [00:00<00:00, 10177.85it/s]
Загружено 5001 предложений
Всего слов для обучения: 75956
Построение словарей...
Обработка слов: 100%|████████████████████████████████████████████████| 75956/75956 [00:00<00:00, 4168837.41it/s]
Всего графем: 393615, уникальных: 34
Всего аллофонов: 391740, уникальных: 58
Топ-10 графем: [('о', 44031), ('е', 31136), ('а', 30657), ('и', 27201), ('н', 24501), ('т', 23793), ('с', 19351), ('в', 17970), ('л', 17395), ('к', 16581)]
Топ-10 аллофонов: [('a1', 21176), ('a4', 20409), ('i1', 18681), ('i4', 18515), ('o0', 17117), ('a0', 16663), ('t', 15992), ('k', 14292), ('j', 14270), ('n', 13938)]
Создание словаря графем: 100%|█████████████████████████████████████████████| 34/34 [00:00<00:00, 1332769.50it/s]
Создание словаря аллофонов: 100%|██████████████████████████████████████████| 58/58 [00:00<00:00, 2079227.62it/s]
Размер словаря графем: 38
Размер словаря аллофонов: 62
Используется устройство: cuda
Всего параметров модели: 2,519,614
Начало обучения...
Epoch 1/30 [Train]: 100%|█████████████████████████| 1899/1899 [00:18<00:00, 105.31it/s, Loss=0.1247, Acc=0.8801]
Epoch 1/30 [Val]: 100%|██████████████████████████████████████| 475/475 [00:01<00:00, 268.48it/s, Val Acc=0.9384]
Epoch 1/30, Loss: 0.3730, Train Acc: 0.8801, Val Acc: 0.9384
Новая лучшая модель сохранена с точностью: 0.9384
Epoch 2/30 [Train]: 100%|█████████████████████████| 1899/1899 [00:17<00:00, 105.86it/s, Loss=0.1252, Acc=0.9463]
Epoch 2/30 [Val]: 100%|██████████████████████████████████████| 475/475 [00:01<00:00, 252.25it/s, Val Acc=0.9532]
Epoch 2/30, Loss: 0.1499, Train Acc: 0.9463, Val Acc: 0.9532
Новая лучшая модель сохранена с точностью: 0.9532
Достигнута высокая точность, останавливаем обучение
Обучение завершено!

============================================================
Тестирование обученной модели
============================================================

Тестирование модели на примерах:
привет: ['p', "r'", 'i1', "v'", 'i1', 't']
мир: ["m'", 'i0', 'r']
тест: ["t'", 'e0', 's', 't']
аллофон: ['a1', 'l', 'l', 'a1', 'f', 'o0', 'n']
графема: ['g', 'r', 'a2', "f'", 'i1', 'm', 'a0']

============================================================
Создание полной фонетической транскрипции
============================================================
Создание полной фонетической транскрипции для 5001 предложений...
Обработка предложений: 100%|███████████████████████████████████████████████| 5001/5001 [00:25<00:00, 194.20it/s]
Полная фонетическая транскрипция сохранена в /home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files/complete_phonetics_result.json
Обработано 5001 предложений, 75956 слов
Создание JSON для сдачи лабораторной работы...
Формирование JSON для сдачи: 100%|█████████████████████████████████████████| 5001/5001 [00:26<00:00, 191.55it/s]
Файл для сдачи лабораторной работы сохранен в /home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files/lab_phonetics_submission.json

Готово! Созданы файлы в директории /home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files:
- best_phonetics_model.pth (обученная модель)
- complete_phonetics_result.json (полная транскрипция всего XML)
- lab_phonetics_submission.json (файл для сдачи лабораторной работы)

Статистика:
- Предложений: 5001
- Слов: 75956
- Аллофонов: 392548

Полные пути к файлам:
- Модель: /home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files/best_phonetics_model.pth
- Полная транскрипция: /home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files/complete_phonetics_result.json
- Для сдачи: /home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files/lab_phonetics_submission.json
(itmo_venv) artyom@artyom-pc:~/itmo$ '''