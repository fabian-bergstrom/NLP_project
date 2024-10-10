from tagger_parser import *
import random

class FixedWindowParserWithBeam(FixedWindowParser):
        
    def beam_search(self, words, tags, beam_size=3): 
        init_config = self.initial_config(len(words))
        init_score = 0.0
        beam = [(init_config, init_score)]
        
        for i in range(len(words)*2 - 1):
            new_beam = []
            
            for config, score in beam:
                valid_moves = self.valid_moves(config)
                features = self.featurize(words, tags, config)
                    
                with torch.no_grad():
                    scores = self.model.forward(features)
                
                scores = F.log_softmax(scores[valid_moves+[ArcStandardParser.ERROR]], dim=0)

                for move_index in range(len(valid_moves)):
                    new_config = self.next_config(config, valid_moves[move_index])
                    new_score = score + scores[move_index].item()
                    new_beam.append((new_config, new_score))
            
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]
            
        best_config, _ = max(beam, key=lambda x: x[1])
        _, _, pred_heads = best_config
        return pred_heads
  

    def predict(self, words, tags, beam_size=3):
        
        # Encode words and tags using the vocabularies
        words = [self.w2i.get(w, UNK_IDX) for w in words]
        tags = [self.t2i.get(t, UNK_IDX) for t in tags]
        
        pred_heads = self.beam_search(words, tags, beam_size)
        return pred_heads



def training_examples_parser_beam(vocab_words, vocab_tags, gold_data, parser, batch_size=100):
    bx = []
    by = []

    for sentence in gold_data:
        # Separate the words, gold tags, and gold heads
        words, tags, gold_heads = zip(*sentence)
        
        # Encode words and tags using the vocabularies
        words = [vocab_words.get(w, UNK_IDX) for w in words]
        tags = [vocab_tags.get(t, UNK_IDX) for t in tags]

        # Call the oracle
        for config, gold_move in oracle_moves(gold_heads):
            bx.append(parser.featurize(words, tags, config))
            by.append(gold_move)
            
            # --- Added error states ---
            valid_moves = parser.valid_moves(config)
            for move in valid_moves:
                if move != gold_move:
                    error_config = parser.next_config(config, move)
                    bx.append(parser.featurize(words, tags, error_config))
                    by.append(parser.ERROR)


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

def train_parser_beam(train_data, n_epochs=1, batch_size=100, lr=1e-2):
    # Create the vocabularies
    vocab_words, vocab_tags = make_vocabs(train_data)

    # Instantiate the parser
    parser = FixedWindowParserWithBeam(vocab_words, vocab_tags)

    # Instantiate the optimizer
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        running_loss = 0
        n_examples = 0
        with tqdm(total=sum(2 * len(s) - 1 for s in train_data)) as pbar:
            for bx, by in training_examples_parser_beam(vocab_words, vocab_tags, train_data, parser, batch_size):
                optimizer.zero_grad()
                output = parser.model.forward(bx)
                loss = F.cross_entropy(output, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_examples += 1
                pbar.set_postfix(loss=running_loss / n_examples)
                pbar.update(len(bx))

    return parser