from tagger_parser import *
from beam_search import *

# --- Swedish Talbanken --- #
train_data_path = 'baseline/talbanken/sv_talbanken_ud_train_projectivize.conllu'
dev_data_path = 'baseline/talbanken/sv_talbanken_ud_dev_projectivize.conllu'

# --- English Web Treebank --- #
#train_data_path = 'baseline/en_ewt-ud-train-projectivized.conllu'
#dev_data_path = 'baseline/en_ewt-ud-dev.conllu'

# --- English big data set --- #
#train_data_path = 'baseline/CoNLL-2009/english/CoNLL2009-ST-English-train-projectivized.txt'
#dev_data_path = 'baseline/CoNLL-2009/english/CoNLL2009-ST-English-development-projectivized.txt'

# --- Acient Hebrew --- #
#train_data_path = 'baseline/hbo_ptnk-ud-train-projectivized.conllu'
#dev_data_path = 'baseline/hbo_ptnk-ud-dev.conllu'



TRAIN_DATA = Treebank(train_data_path)
DEV_DATA = Treebank(dev_data_path)
#TRAIN_DATA[531] 

torch.manual_seed(8)

TAGGER = train_tagger(TRAIN_DATA)
print('{:.4f}'.format(accuracy(TAGGER, DEV_DATA)))


beam_size = 7
#PARSER = train_parser(TRAIN_DATA, n_epochs=1)
PARSER = train_parser_beam(TRAIN_DATA, n_epochs=1)
print("beam size: ", beam_size)
print('{:.4f}'.format(uas(PARSER, DEV_DATA, beam_size)))

print("beam size: ", beam_size)
acc, uas = evaluate(TAGGER, PARSER, DEV_DATA, beam_size)
print('acc: {:.4f}, uas: {:.4f}'.format(acc, uas))