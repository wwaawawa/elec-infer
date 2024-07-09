import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

# 读取数据
df = pd.read_excel('C:/Users/liyongqi/deep-learning/electricityPredict/electricity/structed_data.xlsx')

# # 编码分类特征
# label_encoder = LabelEncoder()
# data['id'] = label_encoder.fit_transform(data['id'])
# data['type'] = label_encoder.fit_transform(data['type'])


#####定义数据集类
class ElectricityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        type_id = row['type']
        house_id = row['id']
        target_sequence = row['target']

        target_text = ' '.join(map(str, target_sequence))
        inputs = self.tokenizer(target_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return input_ids, attention_mask, type_id, house_id

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = ElectricityDataset(df, tokenizer, max_len=50)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#####定义模型，使用BERT编码器和GPT2解码器的组合进行预测
from transformers import BertModel, GPT2LMHeadModel


class BertEncoder(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class GPT2Decoder(nn.Module):
    def __init__(self, pretrained_model_name='gpt2'):
        super(GPT2Decoder, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, encoder_hidden_states, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask)
        return outputs.logits


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, type_embed_dim, num_types):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.type_embedding = nn.Embedding(num_types, type_embed_dim)

    def forward(self, src_input_ids, trg_input_ids, type_ids, src_attention_mask=None, trg_attention_mask=None):
        type_embeddings = self.type_embedding(type_ids).unsqueeze(1).repeat(1, src_input_ids.size(1), 1)
        src_input_ids = torch.cat((src_input_ids, type_embeddings), dim=-1)

        encoder_outputs = self.encoder(src_input_ids, attention_mask=src_attention_mask)
        decoder_outputs = self.decoder(trg_input_ids, encoder_hidden_states=encoder_outputs,
                                       attention_mask=trg_attention_mask)
        return decoder_outputs


# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = BertEncoder().to(device)
decoder = GPT2Decoder().to(device)
model = Seq2Seq(encoder, decoder, type_embed_dim=32, num_types=df['type'].nunique()).to(device)

###训练与评估模型
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

optimizer = Adam(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        src_input_ids, src_attention_mask, type_ids, house_ids = batch
        src_input_ids, src_attention_mask, type_ids = src_input_ids.to(device), src_attention_mask.to(
            device), type_ids.to(device)

        trg_input_ids = src_input_ids.clone()
        trg_input_ids[:, :-10] = tokenizer.pad_token_id

        optimizer.zero_grad()
        output = model(src_input_ids, trg_input_ids, type_ids, src_attention_mask)
        output = output[:, :-1].contiguous().view(-1, output.size(-1))
        trg_input_ids = trg_input_ids[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg_input_ids)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src_input_ids, src_attention_mask, type_ids, house_ids = batch
            src_input_ids, src_attention_mask, type_ids = src_input_ids.to(device), src_attention_mask.to(
                device), type_ids.to(device)

            trg_input_ids = src_input_ids.clone()
            trg_input_ids[:, :-10] = tokenizer.pad_token_id

            output = model(src_input_ids, trg_input_ids, type_ids, src_attention_mask)
            output = output[:, :-1].contiguous().view(-1, output.size(-1))
            trg_input_ids = trg_input_ids[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg_input_ids)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, dataloader, optimizer, criterion, device)
    valid_loss = evaluate(model, dataloader, criterion, device)

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')


# 保存预测结果的函数
def save_predictions(model, dataloader, tokenizer, device, output_file):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            src_input_ids, src_attention_mask, type_ids, house_ids = batch
            src_input_ids, src_attention_mask, type_ids = src_input_ids.to(device), src_attention_mask.to(
                device), type_ids.to(device)

            trg_input_ids = src_input_ids.clone()
            trg_input_ids[:, :-10] = tokenizer.pad_token_id

            output = model(src_input_ids, trg_input_ids, type_ids, src_attention_mask)
            predictions = torch.argmax(output, dim=-1)

            for i in range(predictions.size(0)):
                type_id = type_ids[i].item()
                house_id = house_ids[i]
                target_sequence = predictions[i].cpu().numpy().tolist()

                results.append({
                    'type': type_id,
                    'id': house_id,
                    'predicted_target': target_sequence
                })

    # 创建 DataFrame 并保存到 Excel 文件
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)


# 使用函数进行预测并保存结果
output_file = 'C:/Users/liyongqi/deep-learning/electricityPredict/electricity/predictions.xlsx'
save_predictions(model, dataloader, tokenizer, device, output_file)
print(f"Predictions saved to {output_file}")