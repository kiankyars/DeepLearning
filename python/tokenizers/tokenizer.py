# from bpe import RegexTokenizer
from scratch import RegexTokenizer
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
print(type(list(text.encode("utf-8"))[0])) # <class 'int'>
tokenizer = RegexTokenizer(276, text, GPT4_SPLIT_PATTERN)
tokenizer.save('/Users/kian/Code/DeepLearning/models/tokenizer.pkl')

loaded_tokenizer = RegexTokenizer.load('/Users/kian/Code/DeepLearning/models/tokenizer.pkl')

print(loaded_tokenizer.encode("helloworld"))

print(loaded_tokenizer.decode(loaded_tokenizer.encode("helloworld")))

# print(loaded_tokenizer.merges)