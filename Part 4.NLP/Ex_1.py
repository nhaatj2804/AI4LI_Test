from pyvi import ViTokenizer, ViPosTagger
from pyvi import ViUtils

text = "Trí tuệ nhân tạo đang thay đổi thế giới."
remove_accents = ViUtils.remove_accents(text)

tokens = ViTokenizer.tokenize(text)
pos_tags = ViPosTagger.postagging(text)


print("Token:", tokens)
print("POS Tag:", list(zip(*pos_tags)))
print("Remove Accents:", remove_accents)
