import torch
import torch.nn as nn
import json
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

RESULTS_DIR = "/home/artyom/itmo/syntes_rechi/fixed/phonetic/result_files"
MODEL_PATH = os.path.join(RESULTS_DIR, "best_phonetics_model.pth")

class SimpleGraphemeToAllophoneModel(nn.Module):
    def __init__(self, grapheme_vocab_size, allophone_vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(SimpleGraphemeToAllophoneModel, self).__init__()
        
        self.embedding = nn.Embedding(grapheme_vocab_size, embedding_dim, padding_idx=0)
        
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              batch_first=True, bidirectional=True, dropout=dropout)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, allophone_vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, graphemes, grapheme_mask):
        embedded = self.embedding(graphemes)
        embedded = self.dropout(embedded)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        output = self.output_layer(encoder_outputs)
        return output

def parse_xml_for_inference(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    all_sentences_data = []
    print("Парсинг тестового XML файла...")
    for sentence_idx, sentence in enumerate(tqdm(root.findall('.//sentence'), desc="Обработка предложений")):
        sentence_data = {
            'sentence_id': sentence_idx,
            'words': []
        }
        
        for word in sentence.findall('.//word'):
            original = word.get('original', '')
            if original:
                sentence_data['words'].append({'original': original})
        
        if sentence_data['words']:
            all_sentences_data.append(sentence_data)
    print(f"Загружено {len(all_sentences_data)} предложений из теста")
    return all_sentences_data

def predict_allophones(model, grapheme_to_idx, allophone_to_idx, words):
    device = next(model.parameters()).device
    model.eval()
    idx_to_allophone = {v: k for k, v in allophone_to_idx.items()}
    results = []
    with torch.no_grad():
        for word in words:
            graphemes = [char.lower() for char in word if char.isalpha()]
            if not graphemes:
                results.append([])
                continue
            grapheme_indices = [grapheme_to_idx.get(g, grapheme_to_idx['<UNK>']) for g in graphemes]
            grapheme_tensor = torch.tensor([grapheme_indices], dtype=torch.long).to(device)
            grapheme_mask = torch.ones((1, len(graphemes)), dtype=torch.float).to(device)
            output = model(grapheme_tensor, grapheme_mask)
            _, predicted = torch.max(output, 2)
            predicted_allophones = []
            for idx in predicted[0].cpu().numpy():
                allophone = idx_to_allophone.get(idx, '<UNK>')
                if allophone not in ['<PAD>', '<SOS>', '<EOS>']:
                    predicted_allophones.append(allophone)
            results.append(predicted_allophones)
    return results

def create_inference_json(model, grapheme_to_idx, allophone_to_idx, all_sentences_data, output_file='test_inference_result.json'):
    print(f"Создание фонетической транскрипции для {len(all_sentences_data)} тестовых предложений...")
    result = []
    for sentence_data in tqdm(all_sentences_data, desc="Инференс предложений"):
        sentence_result = {
            "sentence_id": sentence_data['sentence_id'],
            "words": []
        }
        original_words = [word['original'] for word in sentence_data['words']]
        predicted_allophones = predict_allophones(model, grapheme_to_idx, allophone_to_idx, original_words)
        for word_data, allophones in zip(sentence_data['words'], predicted_allophones):
            word_result = {
                "content": word_data['original'],
                "allophones": allophones
            }
            sentence_result["words"].append(word_result)
        result.append(sentence_result)
    output_path = os.path.join(RESULTS_DIR, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Фонетическая транскрипция теста сохранена в {output_path}")
    return result

def load_model_and_dicts(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    grapheme_to_idx = checkpoint['grapheme_to_idx']
    allophone_to_idx = checkpoint['allophone_to_idx']
    model = SimpleGraphemeToAllophoneModel(
        grapheme_vocab_size=len(grapheme_to_idx),
        allophone_vocab_size=len(allophone_to_idx),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, grapheme_to_idx, allophone_to_idx

if __name__ == "__main__":
    print("="*60)
    print("Запуск инференса фонетического транскриптора на тестовом файле")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")
    
    model, grapheme_to_idx, allophone_to_idx = load_model_and_dicts(MODEL_PATH)
    model.to(device)
    
    TEST_XML_PATH = "/home/artyom/itmo/syntes_rechi/fixed/phonetic/Test_Phonetics.xml"
    all_sentences_data = parse_xml_for_inference(TEST_XML_PATH)
    
    create_inference_json(model, grapheme_to_idx, allophone_to_idx, all_sentences_data, output_file="test_inference_result.json")
