import re
import random
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)


SEED = 42
BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 256
EPOCHS = 15
LEARNING_RATE = 0.001
MIN_FREQ = 1

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Appareil utilisé :", device)



def read_sentences(path, preferred_column):
    df = pd.read_csv(path)

    # Utilise la colonne demandée si elle existe,
    # sinon prend la première colonne du fichier.
    column = (
        preferred_column
        if preferred_column in df.columns
        else df.columns[0]
    )

    return df[column].fillna("").astype(str).tolist()


# df = pd.read_csv("dataset.csv") # a donner

df = pd.read_csv(
    "ffr_dataset_v2.txt",
    sep="\t",
    header=None,
    names=["fr", "en"],
    encoding="utf-8",
    on_bad_lines="warn"
)

fr_sentences = df["fr"].fillna("").astype(str).tolist()
en_sentences = df["en"].fillna("").astype(str).tolist()



if len(fr_sentences) != len(en_sentences):
    raise ValueError(
        "Les fichiers FR et EN doivent contenir "
        "le même nombre de lignes alignées."
    )

# Supprimer les paires vides
pairs = [
    (fr.strip(), en.strip())
    for fr, en in zip(fr_sentences, en_sentences)
    if fr.strip() and en.strip()
]

random.shuffle(pairs)

# Répartition entraînement / validation / test
n = len(pairs)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:n_train + n_val]
test_pairs = pairs[n_train + n_val:]

print(f"Entraînement : {len(train_pairs)}")
print(f"Validation   : {len(val_pairs)}")
print(f"Test         : {len(test_pairs)}")


# =====================================================
# 3. TOKENISATION ET VOCABULAIRE
# =====================================================

PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"

SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]


def tokenize(text):
    # Sépare les mots et la ponctuation.
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocabulary:
    def __init__(self, sentences, min_freq=1):
        counter = Counter()

        for sentence in sentences:
            counter.update(tokenize(sentence))

        self.itos = list(SPECIAL_TOKENS)

        for word, freq in counter.items():
            if freq >= min_freq and word not in self.itos:
                self.itos.append(word)

        self.stoi = {
            word: idx for idx, word in enumerate(self.itos)
        }

        self.pad_idx = self.stoi[PAD]
        self.unk_idx = self.stoi[UNK]
        self.sos_idx = self.stoi[SOS]
        self.eos_idx = self.stoi[EOS]

    def encode(self, sentence):
        tokens = tokenize(sentence)
        ids = [
            self.stoi.get(token, self.unk_idx)
            for token in tokens
        ]
        return [self.sos_idx] + ids + [self.eos_idx]

    def decode(self, ids):
        tokens = []

        for idx in ids:
            word = self.itos[idx]

            if word == EOS:
                break

            if word not in SPECIAL_TOKENS:
                tokens.append(word)

        text = " ".join(tokens)

        # Rapproche la ponctuation du mot précédent.
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text


src_vocab = Vocabulary(
    [fr for fr, en in train_pairs], MIN_FREQ
)

tgt_vocab = Vocabulary(
    [en for fr, en in train_pairs], MIN_FREQ
)

print("Vocabulaire français :", len(src_vocab.itos))
print("Vocabulaire anglais  :", len(tgt_vocab.itos))


# =====================================================
# 4. CLASSE DATASET
# =====================================================

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        fr, en = self.pairs[index]

        src_ids = torch.tensor(
            self.src_vocab.encode(fr),
            dtype=torch.long
        )

        tgt_ids = torch.tensor(
            self.tgt_vocab.encode(en),
            dtype=torch.long
        )

        return src_ids, tgt_ids


# =====================================================
# 5. COLLATE FUNCTION : PADDING DES SEQUENCES
# =====================================================

def make_collate_fn(src_vocab, tgt_vocab):
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)

        # Les longueurs sont nécessaires au packing.
        src_lengths = torch.tensor(
            [len(x) for x in src_batch],
            dtype=torch.long
        )

        src_batch = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=src_vocab.pad_idx
        )

        tgt_batch = pad_sequence(
            tgt_batch,
            batch_first=True,
            padding_value=tgt_vocab.pad_idx
        )

        return src_batch, src_lengths, tgt_batch

    return collate_fn


collate_fn = make_collate_fn(src_vocab, tgt_vocab)

train_loader = DataLoader(
    TranslationDataset(train_pairs, src_vocab, tgt_vocab),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    TranslationDataset(val_pairs, src_vocab, tgt_vocab),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# =====================================================
# 6. ENCODEUR LSTM
# =====================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)

        packed = pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, cell) = self.lstm(packed)

        return hidden, cell


# =====================================================
# 7. DECODEUR LSTM
# =====================================================

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_input, hidden, cell):
        embedded = self.embedding(tgt_input)

        outputs, (hidden, cell) = self.lstm(
            embedded, (hidden, cell)
        )

        logits = self.fc(outputs)

        return logits, hidden, cell


# =====================================================
# 8. MODELE SEQ2SEQ
# =====================================================

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, tgt_input):
        hidden, cell = self.encoder(src, src_lengths)

        # Teacher forcing :
        # le décodeur reçoit les tokens attendus précédents.
        logits, _, _ = self.decoder(
            tgt_input, hidden, cell
        )

        return logits


encoder = Encoder(
    len(src_vocab.itos),
    EMBED_DIM,
    HIDDEN_DIM,
    src_vocab.pad_idx
)

decoder = Decoder(
    len(tgt_vocab.itos),
    EMBED_DIM,
    HIDDEN_DIM,
    tgt_vocab.pad_idx
)

model = Seq2Seq(encoder, decoder).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_vocab.pad_idx
)


# =====================================================
# 9. ENTRAINEMENT ET EVALUATION
# =====================================================

def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    total_tokens = 0

    for src, src_lengths, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Entrée du décodeur : SOS + mots précédents
        tgt_input = tgt[:, :-1]

        # Cibles : mots suivants + EOS
        tgt_expected = tgt[:, 1:]

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(
                src, src_lengths, tgt_input
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_expected.reshape(-1)
            )

            if training:
                loss.backward()

                # Stabilise les gradients du LSTM.
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

        valid_tokens = (
            tgt_expected != tgt_vocab.pad_idx
        ).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "src_itos": src_vocab.itos,
                "tgt_itos": tgt_vocab.itos,
                "config": {
                    "embed_dim": EMBED_DIM,
                    "hidden_dim": HIDDEN_DIM
                }
            },
            "seq2seq_fr_en.pt"
        )

print("Modèle sauvegardé dans seq2seq_fr_en.pt")


# =====================================================
# 10. TRADUCTION D'UNE NOUVELLE PHRASE
# =====================================================

@torch.no_grad()
def translate(sentence, max_len=40):
    model.eval()

    src_ids = torch.tensor(
        [src_vocab.encode(sentence)],
        dtype=torch.long,
        device=device
    )

    src_lengths = torch.tensor(
        [src_ids.size(1)], dtype=torch.long
    )

    hidden, cell = model.encoder(src_ids, src_lengths)

    # Le décodeur commence par SOS.
    current = torch.tensor(
        [[tgt_vocab.sos_idx]],
        dtype=torch.long,
        device=device
    )

    generated = []

    for _ in range(max_len):
        logits, hidden, cell = model.decoder(
            current, hidden, cell
        )

        next_id = logits[:, -1, :].argmax(dim=-1).item()

        if next_id == tgt_vocab.eos_idx:
            break

        generated.append(next_id)

        current = torch.tensor(
            [[next_id]], dtype=torch.long, device=device
        )

    return tgt_vocab.decode(generated)


# Recharger le meilleur modèle avant le test
checkpoint = torch.load(
    "seq2seq_fr_en.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

phrase = "Je suis étudiant."
print("Français :", phrase)
print("Anglais  :", translate(phrase))