import conllu
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


PAD = '[PAD]'
UNK = '[UNK]'
PAD_IDX = 0
UNK_IDX = 1


class Treebank(Dataset):

    def __init__(self, filename):
        super().__init__()
        self.items = []
        with open(filename, 'rt', encoding='utf-8') as fp:
            for tokens in conllu.parse_incr(fp):
                sentence = [('[ROOT]', '[ROOT]', 0)]
                for token in tokens.filter(id=lambda x: type(x) is int):
                    sentence.append((token['form'], token['upos'], token['head']))
                self.items.append(sentence)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
    


def make_vocabs(gold_data):
    vocab_words = {PAD: PAD_IDX, UNK: UNK_IDX}
    vocab_tags = {PAD: PAD_IDX}
    for sentence in gold_data:
        for word, tag, _ in sentence:
            if word not in vocab_words:
                vocab_words[word] = len(vocab_words)
            if tag not in vocab_tags:
                vocab_tags[tag] = len(vocab_tags)
    return vocab_words, vocab_tags

class FixedWindowModel(nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim):
        super().__init__()

        # Create the embeddings based on the given specifications
        self.embeddings = nn.ModuleList()
        for n, num_embeddings, embedding_dim in embedding_specs:
            embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
            nn.init.normal_(embedding.weight, std=1e-2)
            for i in range(n):
                self.embeddings.append(embedding)

        # Set up the FFN
        input_dim = sum(e.embedding_dim for e in self.embeddings)
        self.pipe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        embedded = [e(x[..., i]) for i, e in enumerate(self.embeddings)]
        return self.pipe(torch.cat(embedded, -1))
    
class Tagger(object):

    def predict(self, sentence):
        raise NotImplementedError
        
class FixedWindowTagger(Tagger):

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=10, hidden_dim=100):
        embedding_specs = [(3, len(vocab_words), word_dim), (1, len(vocab_tags), tag_dim)]
        self.model = FixedWindowModel(embedding_specs, hidden_dim, len(vocab_tags))
        self.w2i = vocab_words
        self.i2t = {i: t for t, i in vocab_tags.items()}

    def featurize(self, words, i, pred_tags):
        x = torch.zeros(4, dtype=torch.long)
        x[0] = words[i]
        x[1] = words[i - 1] if i > 0 else PAD_IDX
        x[2] = words[i + 1] if i + 1 < len(words) else PAD_IDX
        x[3] = pred_tags[i - 1] if i > 0 else PAD_IDX
        return x

    def predict(self, words):
        words = [self.w2i.get(w, UNK_IDX) for w in words]
        pred_tags = []
        for i in range(len(words)):
            features = self.featurize(words, i, pred_tags)
            with torch.no_grad():
                scores = self.model.forward(features)
            pred_tag = scores.argmax().item()
            pred_tags.append(pred_tag)
        return [self.i2t[i] for i in pred_tags]   

def training_examples_tagger(vocab_words, vocab_tags, gold_data, tagger, batch_size=100, shuffle=False):
    bx = []
    by = []
    for sentence in gold_data:
        # Separate the words and the gold-standard tags
        words, gold_tags, _ = zip(*sentence)

        # Encode words and tags using the vocabularies
        words = [vocab_words.get(w, UNK_IDX) for w in words]
        gold_tags = [vocab_tags[t] for t in gold_tags]

        # Simulate a run of the tagger over the sentence, collecting training examples
        pred_tags = []
        for i, gold_tag in enumerate(gold_tags):
            bx.append(tagger.featurize(words, i, pred_tags))
            by.append(gold_tag)
            if len(bx) >= batch_size:
                bx = torch.stack(bx)
                by = torch.LongTensor(by)
                if shuffle:
                    random_indices = torch.randperm(len(bx))
                    yield bx[random_indices], by[random_indices]
                else:
                    yield bx, by
                bx = []
                by = []
            pred_tags.append(gold_tag)    # teacher forcing!

    # Check whether there is an incomplete batch
    if bx:
        bx = torch.stack(bx)
        by = torch.LongTensor(by)
        if shuffle:
            random_indices = torch.randperm(len(bx))
            yield bx[random_indices], by[random_indices]
        else:
            yield bx, by
            
def train_tagger(train_data, n_epochs=1, batch_size=100, lr=1e-2):
    # Create the vocabularies
    vocab_words, vocab_tags = make_vocabs(train_data)

    # Instantiate the tagger
    tagger = FixedWindowTagger(vocab_words, vocab_tags)

    # Instantiate the optimizer
    optimizer = optim.Adam(tagger.model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        running_loss = 0
        n_examples = 0
        with tqdm(total=sum(len(s) for s in train_data)) as pbar:
            for bx, by in training_examples_tagger(vocab_words, vocab_tags, train_data, tagger):
                optimizer.zero_grad()
                output = tagger.model.forward(bx)
                loss = F.cross_entropy(output, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_examples += 1
                pbar.set_postfix(loss=running_loss/n_examples)
                pbar.update(len(bx))

    return tagger

def accuracy(tagger, gold_data):
    correct = 0
    total = 0
    for sentence in gold_data:
        words, gold_tags, _ = zip(*sentence)
        pred_tags = tagger.predict(words)
        for gold_tag, pred_tag in zip(gold_tags[1:], pred_tags[1:]):  # ignore the pseudo-root
            correct += int(gold_tag == pred_tag)
            total += 1
    return correct / total


class Parser(object):

    def predict(self, words, tags):
        raise NotImplementedError
        
class ArcStandardParser(Parser):

    MOVES = tuple(range(4))

    SH, LA, RA, ERROR = MOVES

    @staticmethod
    def initial_config(num_words):
        return 0, [], [0] * num_words

    @staticmethod
    def valid_moves(config):
        pos, stack, heads = config
        moves = []
        if pos < len(heads):
            moves.append(ArcStandardParser.SH)
        if len(stack) >= 3:  # disallow LA with root as dependent
            moves.append(ArcStandardParser.LA)
        if len(stack) >= 2:
            moves.append(ArcStandardParser.RA)
        return moves

    @staticmethod
    def next_config(config, move):
        pos, stack, heads = config
        stack = list(stack)  # copy because we will modify it
        if move == ArcStandardParser.SH:
            stack.append(pos)
            pos += 1
        else:
            heads = list(heads)  # copy because we will modify it
            s1 = stack.pop()
            s2 = stack.pop()
            if move == ArcStandardParser.LA:
                heads[s2] = s1
                stack.append(s1)
            if move == ArcStandardParser.RA:
                heads[s1] = s2
                stack.append(s2)
        return pos, stack, heads

    @staticmethod
    def is_final_config(config):
        pos, stack, heads = config
        return pos == len(heads) and len(stack) == 1
    
class FixedWindowParser(ArcStandardParser):

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=10, hidden_dim=180):
        embedding_specs = [(3, len(vocab_words), word_dim), (3, len(vocab_tags), tag_dim)]
        self.model = FixedWindowModel(embedding_specs, hidden_dim, len(ArcStandardParser.MOVES))
        self.w2i = vocab_words
        self.t2i = vocab_tags

    def featurize(self, words, tags, config):
        i, stack, heads = config
        x = torch.zeros(6, dtype=torch.long)
        x[0] = words[i] if i < len(words) else PAD_IDX
        x[1] = words[stack[-1]] if len(stack) >= 1 else PAD_IDX
        x[2] = words[stack[-2]] if len(stack) >= 2 else PAD_IDX
        #x[3] = words[stack[-3]] if len(stack) >= 3 else PAD_IDX
        #x[4] = words[stack[-4]] if len(stack) >= 4 else PAD_IDX
        x[3] = tags[i] if i < len(tags) else PAD_IDX
        x[4] = tags[stack[-1]] if len(stack) >= 1 else PAD_IDX
        x[5] = tags[stack[-2]] if len(stack) >= 2 else PAD_IDX
        #x[8] = tags[stack[-3]] if len(stack) >= 3 else PAD_IDX
        #x[9] = tags[stack[-4]] if len(stack) >= 4 else PAD_IDX
        return x

    def predict(self, words, tags , beam_size = 1):
        words = [self.w2i.get(w, UNK_IDX) for w in words]
        tags = [self.t2i.get(t, UNK_IDX) for t in tags]
        config = self.initial_config(len(words))
        valid_moves = self.valid_moves(config)
        while valid_moves:
            features = self.featurize(words, tags, config)
            with torch.no_grad():
                scores = self.model.forward(features)

            # We may only predict valid transitions
            best_score, pred_move = float('-inf'), None
            for move in valid_moves:
                if scores[move] > best_score:
                    best_score, pred_move = scores[move], move

            config = self.next_config(config, pred_move)
            valid_moves = self.valid_moves(config)
        i, stack, pred_heads = config
        return pred_heads    

def oracle_moves(gold_heads):
    # Keep track of how many dependents each head still needs to find
    remaining_count = [0] * len(gold_heads)
    for node in gold_heads:
        remaining_count[node] += 1

    # Simulate a parser
    config = ArcStandardParser.initial_config(len(gold_heads))
    
    while not ArcStandardParser.is_final_config(config):
        pos, stack, heads = config
        if len(stack) >= 2:
            s1 = stack[-1]
            s2 = stack[-2]
            if gold_heads[s2] == s1 and remaining_count[s2] == 0:
                move = ArcStandardParser.LA
                yield config, move
                config = ArcStandardParser.next_config(config, move)
                remaining_count[s1] -= 1
                continue
            if gold_heads[s1] == s2 and remaining_count[s1] == 0:
                move = ArcStandardParser.RA
                yield config, move
                config = ArcStandardParser.next_config(config, move)
                remaining_count[s2] -= 1
                continue
        move = ArcStandardParser.SH
        yield config, move
        config = ArcStandardParser.next_config(config, move)

def training_examples_parser(vocab_words, vocab_tags, gold_data, parser, batch_size=100):
    bx = []
    by = []

    for sentence in gold_data:
        # Separate the words, gold tags, and gold heads
        words, tags, gold_heads = zip(*sentence)

        # Encode words and tags using the vocabularies
        words = [vocab_words.get(w, UNK_IDX) for w in words]
        tags = [vocab_tags[t] for t in tags]

        # Call the oracle
        for config, gold_move in oracle_moves(gold_heads):
            bx.append(parser.featurize(words, tags, config))
            by.append(gold_move)
            if len(bx) >= batch_size:
                bx = torch.stack(bx)
                by = torch.LongTensor(by)
                yield bx, by
                bx = []
                by = []

    # Check whether there is an incomplete batch
    if bx:
        bx = torch.stack(bx)
        by = torch.LongTensor(by)
        yield bx, by

def train_parser(train_data, n_epochs=1, batch_size=100, lr=1e-2):
    # Create the vocabularies
    vocab_words, vocab_tags = make_vocabs(train_data)

    # Instantiate the parser
    parser = FixedWindowParser(vocab_words, vocab_tags)

    # Instantiate the optimizer
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        running_loss = 0
        n_examples = 0
        with tqdm(total=sum(2*len(s)-1 for s in train_data)) as pbar:
            for bx, by in training_examples_parser(vocab_words, vocab_tags, train_data, parser):
                optimizer.zero_grad()
                output = parser.model.forward(bx)
                loss = F.cross_entropy(output, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_examples += 1
                pbar.set_postfix(loss=running_loss/n_examples)
                pbar.update(len(bx))

    return parser

def uas(parser, gold_sentences, beam_size = 3):
    correct = 0
    total = 0
    for sentence in gold_sentences:
        words, tags, gold_heads = zip(*sentence)
        pred_heads = parser.predict(words, tags, beam_size=beam_size)
        for gold, pred in zip(gold_heads[1:], pred_heads[1:]):  # ignore the pseudo-root
            correct += int(gold == pred)
            total += 1
    return correct / total

def evaluate(tagger, parser, gold_sentences, beam_size = 3):
    correct_tagger = 0
    total_tagger = 0
    correct_parser = 0
    total_parser = 0
    for sentence in gold_sentences:
        words, gold_tags, gold_heads = zip(*sentence)
        pred_tags = tagger.predict(words)
        for gold, pred in zip(gold_tags[1:], pred_tags[1:]):
            correct_tagger += int(gold == pred)
            total_tagger += 1
        pred_heads = parser.predict(words, pred_tags, beam_size)
        for gold, pred in zip(gold_heads[1:], pred_heads[1:]):
            correct_parser += int(gold == pred)
            total_parser += 1
    return correct_tagger / total_tagger, correct_parser / total_parser