from bpe import RegexTokenizer

loaded_tokenizer = RegexTokenizer.load('/Users/kian/Code/DeepLearning/models/tokenizer.pkl')

print(loaded_tokenizer)

print(loaded_tokenizer.decode(loaded_tokenizer.encode("helloworld")))