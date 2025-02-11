import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from transformers import Qwen2ForCausalLM, GPTNeoForCausalLM, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence
from torch.cuda import amp
import math
import os
import sys

# ==========================
# 1) CrossAttentionLayer (базовая версия)
# ==========================
class CrossAttentionLayer(nn.Module):
    def __init__(self, qwen_dim, gptneo_dim, num_heads=8):
        super().__init__()
        
        assert qwen_dim % num_heads == 0, "qwen_dim must be divisible by num_heads"
        assert gptneo_dim % num_heads == 0, "gptneo_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.qwen_dim = qwen_dim  # напр. 1536
        self.gptneo_dim = gptneo_dim      # напр. 768
        self.head_dim = gptneo_dim // num_heads
        
        # Q, K, V проекции
        self.q_proj = nn.Linear(gptneo_dim, gptneo_dim)      # 768 -> 768
        self.k_proj = nn.Linear(qwen_dim, gptneo_dim)    # 1536 -> 768
        self.v_proj = nn.Linear(qwen_dim, gptneo_dim)    # 1536 -> 768
        
        self.out_proj = nn.Linear(gptneo_dim, gptneo_dim)    # 768 -> 768
        
        self.norm1 = nn.LayerNorm(qwen_dim)  # 1536
        self.norm2 = nn.LayerNorm(gptneo_dim)    # 768
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, qwen_states, gptneo_states, attention_mask=None):
        """
        qwen_states: [batch_size, seq_len, 1536]
        gptneo_states:   [batch_size, seq_len, 768]
        """
        batch_size = qwen_states.size(0)
        
        # Нормализация входов
        qwen_states = self.norm1(qwen_states)
        gptneo_states = self.norm2(gptneo_states)
        
        # Q, K, V
        q = self.q_proj(gptneo_states)          # -> [batch_size, seq_len, 768]
        k = self.k_proj(qwen_states)        # -> [batch_size, seq_len, 768]
        v = self.v_proj(qwen_states)        # -> [batch_size, seq_len, 768]
        
        # Разделение на головы
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Маска (если есть)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax и dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Применяем attention к v
        context = torch.matmul(attn, v)
        
        # Склеиваем обратно
        context = context.transpose(1, 2).contiguous()  # [batch, seq, heads, head_dim]
        context = context.view(batch_size, -1, self.gptneo_dim)  # -> [batch, seq, 768]
        
        output = self.out_proj(context)  # -> [batch, seq, 768]
        return output

# ==========================
# 1.1) CrossAttentionAdapter
# ==========================
class CrossAttentionAdapter(nn.Module):
    """
    Небольшой feed-forward блок, позволяющий адаптировать выход cross-attention.
    """
    def __init__(self, hidden_dim, adapter_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(adapter_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x + residual

# ==========================
# 1.2) EnhancedCrossAttentionLayer
# ==========================
class EnhancedCrossAttentionLayer(nn.Module):
    def __init__(self, qwen_dim, gptneo_dim, num_heads=8, adapter_dim=256):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(qwen_dim, gptneo_dim, num_heads)
        self.adapter = CrossAttentionAdapter(gptneo_dim, adapter_dim)
        self.gate = nn.Linear(gptneo_dim * 2, gptneo_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, qwen_states, gptneo_states, attention_mask=None):
        # Вычисляем стандартное cross-attention
        attn_out = self.cross_attn(qwen_states, gptneo_states, attention_mask)
        # Пропускаем через адаптационный блок
        attn_out = self.adapter(attn_out)
        # Вычисляем веса гейта на основе конкатенации исходных представлений и результата cross-attention
        combined = torch.cat([gptneo_states, attn_out], dim=-1)
        gate_weights = self.sigmoid(self.gate(combined))
        # Динамически смешиваем: если gate_weights близки к 1 – сильно влияют внешние знания,
        # если к 0 – остаётся исходное представление.
        output = gate_weights * attn_out + (1 - gate_weights) * gptneo_states
        return output

# ==========================
# 2) ModifiedQwenWithCrossAttention
# ==========================
class ModifiedQwenWithCrossAttention(nn.Module):
    def __init__(self, qwen_config, gptneo_config):
        super().__init__()
        
        print("Loading pretrained Qwen weights...")
        self.model = Qwen2ForCausalLM.from_pretrained(
            "Qwen/Qwen2-1.5B",
            config=qwen_config,
            trust_remote_code=True
        )
        
        print("Freezing Qwen weights...")
        for param in self.model.parameters():
            param.requires_grad = False
        
        # В Qwen2 (Qwen) hidden_size = 1536
        # "pre_proj" — легкая линейная проекция 1536->1536
        print("Initializing projection layers...")
        self.pre_proj = nn.Linear(qwen_config.hidden_size, qwen_config.hidden_size)
        
        # Проекция в размер GPT-Neo (768)
        self.proj = nn.Linear(qwen_config.hidden_size, gptneo_config.hidden_size)
        
        print("Initializing enhanced cross-attention layers...")
        # Вместо стандартных слоёв заменяем на улучшенные (с адаптером и gating)
        self.cross_attention_layers = nn.ModuleList([
            EnhancedCrossAttentionLayer(
                qwen_dim=qwen_config.hidden_size,  # 1536
                gptneo_dim=gptneo_config.hidden_size,       # 768
                num_heads=8,
                adapter_dim=256
            )
            for _ in range(2)
        ])
        
        # Промежуточный feed-forward: [1536 -> 1152 -> 768]
        intermediate_dim = (qwen_config.hidden_size + gptneo_config.hidden_size) // 2  # (1536+768)//2 = 1152
        self.intermediate_layers = nn.Sequential(
            nn.Linear(qwen_config.hidden_size, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, gptneo_config.hidden_size),
            nn.LayerNorm(gptneo_config.hidden_size)
        )

    def forward(self, input_ids, attention_mask=None):
        # Прогон через Qwen без lm_head
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
        # Последнее скрытое состояние [batch, seq_len, 1536]
        hidden_states = outputs.hidden_states[-1]
        
        # Лёгкая проекция 1536 -> 1536
        pre_projected_states = self.pre_proj(hidden_states)
        
        # Проекция до размера GPT-Neo (768)
        projected_states = self.proj(pre_projected_states)  # [batch, seq, 768]
        
        # Получаем промежуточное представление через FFN (1536->768)
        intermediate_states = self.intermediate_layers(pre_projected_states)  # [batch, seq, 768]
        
        # Инициализируем gptneo_states как intermediate_states
        gptneo_states = intermediate_states
        
        # Пропускаем через серию улучшенных cross-attention слоёв
        for layer in self.cross_attention_layers:
            cross_attn_output = layer(pre_projected_states, gptneo_states, attention_mask)
            gptneo_states = gptneo_states + cross_attn_output
        
        # Финальное объединение: складываем gptneo_states и projected_states
        final_states = gptneo_states + projected_states
        return final_states

# ==========================
# 3) ModifiedGptNeo
# ==========================
class ModifiedGptNeo(nn.Module):
    def __init__(self, gptneo_config, qwen_tokenizer):
        super().__init__()
        
        self.base_model = GPTNeoForCausalLM(gptneo_config)
        
        # Замораживаем embedding слои
        self.base_model.transformer.wte.requires_grad = False
        self.base_model.transformer.wpe.requires_grad = False
        
        # Зануляем позиционные эмбеддинги, чтобы они не добавлялись к входным представлениям
        self.base_model.transformer.wpe.weight.data.zero_()
        
        # Новый lm_head под словарь Qwen
        qwen_vocab_size = len(qwen_tokenizer)
        hidden_size = gptneo_config.hidden_size
        self.new_lm_head = nn.Linear(hidden_size, qwen_vocab_size, bias=False)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GPT-Neo trainable parameters: {trainable_params}")

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [batch, seq, 768]
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        device = hidden_states.device

        with torch.cuda.amp.autocast():
            max_pos_length = self.base_model.transformer.wpe.num_embeddings
            if sequence_length > max_pos_length:
                hidden_states = hidden_states[:, :max_pos_length, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :max_pos_length]
                sequence_length = max_pos_length
            # Генерируем position_ids
            position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
            
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :]

            hidden_states = hidden_states.contiguous()
            
            # Передаем inputs_embeds и position_ids
            transformer_outputs = self.base_model.transformer(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )

            del hidden_states, attention_mask, position_ids
            torch.cuda.empty_cache()
            
            return self.new_lm_head(transformer_outputs.last_hidden_state)

# ==========================
# 4) CombinedModel
# ==========================
class CombinedModel(nn.Module):
    def __init__(self, qwen_config, gptneo_config, qwen_tokenizer):
        super().__init__()

        print("Initializing CombinedModel...")
        self.qwen = ModifiedQwenWithCrossAttention(
            qwen_config=qwen_config,
            gptneo_config=gptneo_config
        )
        
        self.gptneo = ModifiedGptNeo(
            gptneo_config=gptneo_config,
            qwen_tokenizer=qwen_tokenizer
        )

        self.qwen.model.gradient_checkpointing_enable()
        self.gptneo.base_model.gradient_checkpointing_enable()
        
        # Выводим информацию о параметрах
        qwen_params = sum(p.numel() for p in self.qwen.model.parameters())
        print(f"Qwen total parameters: {qwen_params}")
        qwen_trainable = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
        print(f"Qwen trainable parameters: {qwen_trainable}")

    def forward(self, input_ids, attention_mask=None):
        # Получаем промежуточное представление из Qwen
        gptneo_states = self.qwen(input_ids, attention_mask)
        # Получаем логиты из GPT-Neo
        logits = self.gptneo(gptneo_states, attention_mask)
        return logits

# ==========================
# 5) Создание модели и оптимизатора
# ==========================
def create_model_and_optimizer():
    print("Creating model...")
    
    # Загружаем конфигурации
    qwen_config = AutoConfig.from_pretrained(
        "Qwen/Qwen2-1.5B",
        trust_remote_code=True
    )
    gptneo_config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    
    # Загружаем qwen_tokenizer (устанавливаем pad_token, если отсутствует)
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-1.5B",
        trust_remote_code=True
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id
    
    print(f"Qwen hidden size: {qwen_config.hidden_size}")
    print(f"GPT-Neo hidden size: {gptneo_config.hidden_size}")
    print(f"Qwen vocab size: {len(qwen_tokenizer)}")
    
    # Создаем комбинированную модель
    model = CombinedModel(qwen_config, gptneo_config, qwen_tokenizer)
    
    # Настраиваем оптимизатор: обновляются слои cross-attention, intermediate, pre_proj, proj, и GPT-Neo часть
    optimizer = AdamW([
        {'params': model.qwen.cross_attention_layers.parameters(), 'lr': 1e-4},
        {'params': model.qwen.intermediate_layers.parameters(), 'lr': 1e-4},
        {'params': model.qwen.pre_proj.parameters(), 'lr': 1e-4},
        {'params': model.qwen.proj.parameters(), 'lr': 1e-4},
        {'params': [p for p in model.gptneo.parameters() if p.requires_grad], 'lr': 5e-5}
    ])
    
    print("Model and optimizer created successfully")
    return model, optimizer, qwen_tokenizer

# ==========================
# 6) Ранняя остановка
# ==========================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==========================
# 7) Dataset
# ==========================
class BespokeDataset(Dataset):
    def __init__(self, hf_dataset, qwen_tokenizer, max_length=None, split="train"):
        self.dataset = hf_dataset[split]
        self.qwen_tokenizer = qwen_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        conversations = item['conversations']

        system_message = (
            "System:\n\n**Your role as an assistant involves thoroughly exploring questions through a systematic "
            "long thinking process before providing the final precise and accurate solutions. This requires "
            "engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, "
            "backtracing, and iteration to develop well-considered thinking process. Please structure your "
            "response into two main sections: Thought and Solution. In the Thought section, detail your reasoning "
            "process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} "
            "<|end_of_thought|> Each step should include detailed considerations such as analisying questions, "
            "summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, "
            "refining any errors, and revisiting previous steps. In the Solution section, based on various "
            "attempts, explorations, and reflections from the Thought section, systematically present the final "
            "solution that you deem correct. The solution should remain a logical, accurate, concise expression "
            "style and detail necessary step needed to reach the conclusion, formatted as follows: "
            "<|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> "
            "Now, try to solve the following question through the above guidelines:**"
        )

        full_text = f"{system_message}\n\n"
        for conv in conversations:
            if conv['from'] == 'user':
                full_text += f"User:\n\n{conv['value']}\n\n"
            elif conv['from'] == 'assistant':
                full_text += f"Assistant:\n\n{conv['value']}\n\n"

        if self.max_length is not None:
            encoding = self.qwen_tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        else:
            encoding = self.qwen_tokenizer(
                full_text,
                truncation=True,
                return_tensors='pt'
            )

        # Преобразуем полученные тензоры в списки чисел
        input_ids = encoding["input_ids"].squeeze(0).tolist()
        attention_mask = encoding["attention_mask"].squeeze(0).tolist()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # labels совпадают с input_ids
            "raw_text": full_text
        }

# Функция collate_fn для динамического паддинга с использованием pad_sequence
class CollateWrapper:
    def __init__(self, qwen_tokenizer):
        self.qwen_tokenizer = qwen_tokenizer

    def __call__(self, batch: List[Dict]) -> Dict:
        # Преобразуем каждую последовательность (input_ids, attention_mask, labels) в тензор
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        attention_masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        raw_texts = [item["raw_text"] for item in batch]

        # Дополняем последовательности до длины самого длинного примера в батче
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.qwen_tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        # Для labels можно использовать значение -100 для игнорирования при расчете loss
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels,
            "raw_text": raw_texts
        }

def prepare_dataset(batch_size=4, val_split=0.1, subset_size=0.1, epoch_seed=None, max_length=4096):
    print("Loading dataset...")
    try:
        dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        raise

    print("Loading tokenizers...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-1.5B",
        trust_remote_code=True
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id

    if epoch_seed is not None:
        full_dataset = dataset['train'].shuffle(seed=epoch_seed)
    else:
        full_dataset = dataset['train'].shuffle(seed=42)

    subset_size = int(len(full_dataset) * subset_size)
    subset_dataset = full_dataset.select(range(subset_size))
    print(f"Using {subset_size} examples out of {len(full_dataset)} total")

    # Фильтрация примеров по длине (не обрезаем, а исключаем слишком длинные)
    def filter_example(example):
        # Собираем полный текст, аналогично __getitem__ класса BespokeDataset
        system_message = (
            "System:\n\n**Your role as an assistant involves thoroughly exploring questions through a systematic "
            "long thinking process before providing the final precise and accurate solutions. This requires "
            "engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, "
            "backtracing, and iteration to develop well-considered thinking process. Please structure your "
            "response into two main sections: Thought and Solution. In the Thought section, detail your reasoning "
            "process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} "
            "<|end_of_thought|> Each step should include detailed considerations such as analisying questions, "
            "summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, "
            "refining any errors, and revisiting previous steps. In the Solution section, based on various "
            "attempts, explorations, and reflections from the Thought section, systematically present the final "
            "solution that you deem correct. The solution should remain a logical, accurate, concise expression "
            "style and detail necessary step needed to reach the conclusion, formatted as follows: "
            "<|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> "
            "Now, try to solve the following question through the above guidelines:**"
        )
        full_text = f"{system_message}\n\n"
        for conv in example['conversations']:
            if conv['from'] == 'user':
                full_text += f"User:\n\n{conv['value']}\n\n"
            elif conv['from'] == 'assistant':
                full_text += f"Assistant:\n\n{conv['value']}\n\n"
        # Токенизируем без обрезания, чтобы узнать полную длину
        encoding = qwen_tokenizer(full_text, truncation=False)
        # Если длина превышает max_length – исключаем пример
        return len(encoding["input_ids"]) <= max_length

    print("Filtering dataset to remove examples exceeding maximum length...")
    subset_dataset = subset_dataset.filter(filter_example)
    print(f"After filtering: {len(subset_dataset)} examples remain out of {subset_size} initially selected.")

    # # Остановка для проверки логов – обучение не продолжается
    # print("Execution stopped after filtering. Please check the log above to see how many examples remain.")
    # sys.exit(0)

    # Дальнейшее разделение на тренировочную и валидационную выборки
    val_size = int(len(subset_dataset) * val_split)
    train_size = len(subset_dataset) - val_size

    train_dataset = BespokeDataset(
        {'train': subset_dataset.select(range(train_size))},
        qwen_tokenizer,
        max_length=None,
        split="train"
    )
    val_dataset = BespokeDataset(
        {'train': subset_dataset.select(range(train_size, len(subset_dataset)))},
        qwen_tokenizer,
        max_length=None,
        split="train"
    )

    custom_collate_fn = CollateWrapper(qwen_tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, qwen_tokenizer


def optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

# ==========================
# 9) train_model
# ==========================
def train_model(model, optimizer, epochs=50, batch_size=4, learning_rate=1e-4, 
                max_grad_norm=1.0, device='cuda', debug=False,
                patience=5, min_delta=1e-5, subset_size=0.3):
    print("Initializing training...")
    torch.cuda.empty_cache()
    scaler = torch.cuda.amp.GradScaler()
    
    checkpoint_path = 'best_model_checkpoint.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_state_to_device(optimizer, device)
        initial_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Loaded checkpoint with validation loss: {initial_val_loss:.4f}")
        best_val_loss = initial_val_loss
    else:
        print("No existing checkpoint found. Starting training from scratch.")
        best_val_loss = float('inf')
    
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    train_loader, val_loader, qwen_tokenizer = None, None, None
    last_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        torch.cuda.empty_cache()

        if epoch == 0 or last_val_loss < 1.8:
            train_loader, val_loader, qwen_tokenizer = prepare_dataset(
                batch_size=batch_size,
                subset_size=subset_size,
                epoch_seed=epoch
            )

        total_loss = 0
        model.train()
        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            torch.cuda.empty_cache()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if debug and batch_idx % 100 == 0:
                for i in range(input_ids.size(0)):
                    decoded_text = qwen_tokenizer.decode(input_ids[i].detach().cpu().tolist(), skip_special_tokens=False)
                    eos_token = qwen_tokenizer.eos_token
                    if eos_token in decoded_text:
                        decoded_text = decoded_text.split(eos_token)[0] + eos_token
                    print(f"\n[DEBUG] Decoded text for sample {i} in batch {batch_idx}:")
                    print(decoded_text)
                    print("------------------------------------------------------")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                # Усечение меток: если модель возвращает выходы длины L,
                # используем первые L токенов меток для выравнивания.
                truncated_seq_len = outputs.shape[1]  # БЕЗ +1!
                labels_trunc = labels[:, :truncated_seq_len]
                
                vocab_size = len(qwen_tokenizer)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = labels_trunc[..., 1:].contiguous()

                pad_token_id = qwen_tokenizer.pad_token_id
                shift_labels = torch.where(
                    shift_labels == pad_token_id,
                    torch.tensor(-100, device=shift_labels.device),
                    shift_labels
                )

                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='mean'
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            if debug and (batch_idx % 100 == 0):
                i = 0
                seq_len = input_ids.size(1)
                last_k = 150 if seq_len > 150 else seq_len - 1
                context_ids = input_ids[i, max(0, seq_len - 1 - last_k): seq_len - 1]
                context_text = qwen_tokenizer.decode(context_ids.tolist(), skip_special_tokens=False)
                gold_id = shift_labels[i, -1].item()
                gold_label_text = qwen_tokenizer.decode([gold_id], skip_special_tokens=False) if gold_id != -100 else "<PAD or no-label>"
                logits_for_last_pos = shift_logits[i, -1, :]
                predicted_id = logits_for_last_pos.argmax(dim=-1).item()
                predicted_text = qwen_tokenizer.decode([predicted_id], skip_special_tokens=False)
                top_k = 5
                top_values, top_indices = torch.topk(logits_for_last_pos, k=top_k)
                top_tokens_str = ", ".join(
                    [f"'{qwen_tokenizer.decode([idx.item()], skip_special_tokens=False)}'({val.item():.2f})" 
                     for val, idx in zip(top_values, top_indices)]
                )
                
                print("\n[DEBUG] ============ BATCH DEBUG INFO ============")
                print(f"Epoch {epoch+1}, batch_idx = {batch_idx}")
                print(f"Context (last {last_k} tokens *excluding* final):\n  {context_text}")
                print(f"Gold label (next token):\n  {gold_label_text}")
                print(f"Predicted token:\n  {predicted_text}")
                print(f"Top-{top_k} tokens by logit:\n  {top_tokens_str}")
                print("============================================\n")
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nAverage training loss: {avg_loss:.4f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                truncated_seq_len = outputs.shape[1]
                labels_trunc = labels[:, :truncated_seq_len]
                
                vocab_size = len(qwen_tokenizer)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = labels_trunc[..., 1:].contiguous()
                
                pad_token_id = qwen_tokenizer.pad_token_id
                shift_labels = torch.where(
                    shift_labels == pad_token_id,
                    torch.tensor(-100, device=shift_labels.device),
                    shift_labels
                )
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='mean'
                )

                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        last_val_loss = avg_val_loss
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving new best model with validation loss: {avg_val_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        model.train()

    return best_val_loss

# ==========================
# 10) load_model_for_inference
# ==========================
def load_model_for_inference():
    print("Initializing model for inference...")
    
    qwen_config = AutoConfig.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
    gptneo_config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id
    
    model = CombinedModel(qwen_config, gptneo_config, qwen_tokenizer)
    
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters before loading: {trainable_params_before}")
    
    checkpoint_loaded = False
    if os.path.exists('model_checkpoint.pth'):
        print("Loading model_checkpoint.pth...")
        try:
            checkpoint = torch.load('model_checkpoint.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_loaded = True
            print("Successfully loaded model_checkpoint.pth")
        except Exception as e:
            print(f"Error loading model_checkpoint.pth: {str(e)}")
    
    if not checkpoint_loaded and os.path.exists('best_model_checkpoint.pth'):
        print("Loading best_model_checkpoint.pth...")
        try:
            checkpoint = torch.load('best_model_checkpoint.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_loaded = True
            print("Successfully loaded best_model_checkpoint.pth")
        except Exception as e:
            print(f"Error loading best_model_checkpoint.pth: {str(e)}")
    
    if not checkpoint_loaded:
        raise FileNotFoundError("No checkpoint files found or failed to load checkpoints!")
        
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after loading: {trainable_params_after}")
    
    print("\nChecking cross-attention layers:")
    for i, layer in enumerate(model.qwen.cross_attention_layers):
        q_weight_stats = layer.cross_attn.q_proj.weight.data
        print(f"Layer {i} Q projection stats: mean={q_weight_stats.mean():.4f}, std={q_weight_stats.std():.4f}")
    
    return model, qwen_tokenizer

# ==========================
# 11) generate_response
# ==========================
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.95,
    filter_value: float = -float('Inf')
):
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_values, torch.tensor(filter_value, device=logits.device), logits)
    
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        indices_to_remove = sorted_indices[sorted_mask]
        logits[0, indices_to_remove] = filter_value
    
    return logits

def generate_response(prompt, max_length=4096, temperature=0.9, top_k=100, top_p=0.98, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    try:
        model, qwen_tokenizer = load_model_for_inference()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model = model.to(device)
    model.eval()

    system_message = (
        "System:\n\n**Your role as an assistant involves thoroughly exploring questions through a systematic "
        "long thinking process before providing the final precise and accurate solutions. This requires "
        "engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, "
        "backtracing, and iteration to develop well-considered thinking process. Please structure your "
        "response into two main sections: Thought and Solution. In the Thought section, detail your reasoning "
        "process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} "
        "<|end_of_thought|> Each step should include detailed considerations such as analisying questions, "
        "summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, "
        "refining any errors, and revisiting previous steps. In the Solution section, based on various "
        "attempts, explorations, and reflections from the Thought section, systematically present the final "
        "solution that you deem correct. The solution should remain a logical, accurate, concise expression "
        "style and detail necessary step needed to reach the conclusion, formatted as follows: "
        "<|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> "
        "Now, try to solve the following question through the above guidelines:**"
    )
    
    formatted_prompt = f"{system_message}\n\nUser:\n\n{prompt}\n\nAssistant:\n\n"
    print(f"\nFormatted prompt: {formatted_prompt}")
    
    inputs = qwen_tokenizer(
        formatted_prompt, 
        return_tensors='pt',
        truncation=True,
        max_length=None,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"\nInput tokens: {qwen_tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")
    
    current_input_ids = inputs['input_ids']
    current_attention_mask = inputs['attention_mask']
    
    generated_tokens = []
    min_length = 20  # минимальное число токенов перед остановкой

    # Потоковый вывод: каждый токен печатается сразу при генерации.
    for _ in range(max_length):
        outputs = model(current_input_ids, current_attention_mask)
        next_token_logits = outputs[:, -1, :]
        next_token_logits = next_token_logits / temperature
        next_token_logits = top_k_top_p_filtering(
            next_token_logits,
            top_k=top_k,
            top_p=top_p,
            filter_value=-float('Inf')
        )
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        generated_tokens.append(token_id)
        
        # Декодируем только последний токен для вывода
        decoded_token = qwen_tokenizer.decode([token_id], skip_special_tokens=True)
        sys.stdout.write(decoded_token)
        sys.stdout.flush()
        
        # Проверяем условие завершения генерации
        current_generated_text = qwen_tokenizer.decode(generated_tokens, skip_special_tokens=False)
        if (token_id == qwen_tokenizer.eos_token_id and current_input_ids.shape[1] >= min_length) or ("<|end_of_solution|>" in current_generated_text):
            break
        
        current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
        new_mask = torch.ones((current_attention_mask.size(0), 1), device=device, dtype=current_attention_mask.dtype)
        current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
    
    print("\n---- END ----")
    
    response = qwen_tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return response.strip()

# ==========================
# 12) interactive_chat
# ==========================
def interactive_chat():
    print("\nStarting interactive chat session...")
    print("Type 'exit' to end the conversation\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("\nEnding chat session...")
            break
            
        if not user_input:
            continue
            
        print("\nGenerating response...")
        response = generate_response(user_input)
        
        if response:
            print(f"\nAssistant:\n>>>{response}")
        else:
            print("\nError: Failed to generate response")

# ==========================
# 13) main()
# ==========================
def main():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" 
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.max_memory_allocated = lambda device=None: 0
    # torch.multiprocessing.set_sharing_strategy('file_system')
    
    model, optimizer, qwen_tokenizer = create_model_and_optimizer()
    
    best_val_loss = train_model(
        model=model,
        optimizer=optimizer,
        epochs=15,
        batch_size=1,
        patience=5,
        min_delta=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        debug=True,
        subset_size=0.3
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'model_checkpoint.pth')
    
    print(f"\nTraining complete. Best val loss: {best_val_loss}")

    # interactive_chat()

if __name__ == "__main__":
    main()
