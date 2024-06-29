#%%
import re
from collections import defaultdict, Counter
from tqdm import tqdm

#%%
def get_stats(vocab):
    """
    Given a vocabulary (dictionary mapping words to frequency counts), returns a 
    dictionary of tuples representing the frequency count of pairs of characters 
    in the vocabulary.
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    Given a pair of characters and a vocabulary, returns a new vocabulary with the 
    pair of characters merged together wherever they appear.
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_vocab(inputfiles, vocab):
    """
    Given a list of strings, returns a dictionary of words mapping to their frequency 
    count in the data.
    """
    for inputfile in inputfiles:
        with open(inputfile, "r") as f:
            for line in f:
                for word in line.split():
                    vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab


def byte_pair_encoding(inputfiles, vocab, n):
    """
    Given a list of strings and an integer n, returns a list of n merged pairs
    of characters found in the vocabulary of the input data.
    """
    get_vocab(inputfiles, vocab)
    print(vocab)
    for i in tqdm(range(n)):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab

def modify_dict(dict):
    nd=defaultdict(int)
    for token, freq in dict.items():
        nt=token.replace(' ', '').replace("</w>", '')
        nd[nt]=freq
        # del dict[token]
    del dict
    return nd

# Example usage:
corpus = ["corpus\mg1.txt" , "corpus\mg2.txt",  "corpus\mg3.txt",  "corpus\mg4.txt",  "corpus\mg5.txt",  "corpus\mg6.txt"]
#%%
vocab = defaultdict(int)

vocab_count = 11000
bpe_pairs = byte_pair_encoding(corpus, vocab, vocab_count)
bpe_pairs=modify_dict(bpe_pairs)
print(bpe_pairs)


# %%
def tokenize(word):
    tokens=[]
    i=0
    while(i<len(word)):
        matched=False
        for j in range(len(word), i, -1):
            subtoken=word[i:j]
            if subtoken in bpe_pairs:
                tokens.append(subtoken)
                i=j
                matched=True
                break
        if not matched:
            tokens.append("<UNK>")
            bpe_pairs["<UNK>"]+=1
            i+=1
    return tokens

#%%

N=4


def tokenize_corpus(corpus):
    with open(corpus, "r") as f:
        lines=f.readlines()
    # tokenized_corpus=[]
    # SOS="<s> "*max(1, N-1)
    # t=max(1,N-1)
    # EOS=" </s>"
    with open(corpus, "w") as f:
        for line in lines:
            # tokenized_sent=[SOS]
            tokenized_sent=[]
            words=line.split()
            for word in words:
                tokens=tokenize(word)
                tokenized_sent.extend(tokens)
            # sent=' '.join(tokenized_sent)+EOS+'\n'
            sent=' '.join(tokenized_sent)+'\n'
            # bpe_pairs[SOS]+=t
            # bpe_pairs[EOS]+=1
            f.write(sent)
    print(corpus, "tokenized")
    return



# %%
c="corpus\mg"
for i in range(98):
    tokenize_corpus(c+str(i+1)+".txt")

# %%
def read_corpus(file_path):
    with open(file_path, 'r') as file:
        corpus = [line.strip().split() for line in file.readlines()]
    return corpus

corpus=[]

for i in range(98):
    corpus.extend(read_corpus(c+str(i+1)+".txt"))
    print(c+str(i+1)+".txt read")

# %%
def n_gram_maker(corpus, n):
    n_grams=[]
    for i in tqdm(range(len(corpus))):
        line=corpus[i]
        line=["<s>"]*max(n-1, 0)+line+["</s>"]
        for i in range(len(line)-n+1):
            n_grams.append(tuple(line[i:i+n]))
    return n_grams


# %%
def calculate_counts(ngrams, n):
    counts = defaultdict(Counter)
    lower_counts = Counter()
    for i in tqdm(range(len(ngrams))):
        ngram=ngrams[i]
        counts[ngram[:-1]][ngram[-1]] += 1
        # if n > 1:
        #     lower_counts[ngram[1:]] += 1
    return counts, lower_counts

#%%
def unique_counts(n_grams):
    uc=defaultdict(set)
    for n_gram in n_grams:
        context=n_gram[:-1]
        word=n_gram[-1]
        uc[context].add(word)
    follow_counts=defaultdict(int)
    for context, words in uc.items():
        follow_counts[context]=len(words)
    return follow_counts


# %%
def kneser_ney_smoothing(counts, unique_follow_counts, lower_order_counts, n, D=0.75):
    kn_prob = defaultdict(float)

    for ngram, count_dict in counts.items():
        for word, count in count_dict.items():
            print(ngram, word)
            continuation_count = unique_follow_counts[ngram[1:]]
            if n > 1:
                lower_order_ngram = ngram[1:]
                lower_order_prob = kneser_ney_smoothing(
                    {lower_order_ngram: counts[lower_order_ngram]}, 
                    unique_follow_counts, lower_order_counts, n-1, D
                )[lower_order_ngram]
            else:
                lower_order_prob = continuation_count / sum(unique_follow_counts.values())

            discount = max(count - D, 0) / sum(count_dict.values())
            
            lambda_factor = (D / sum(count_dict.values())) * len(count_dict)

            kn_prob[ngram + (word,)] = discount + lambda_factor * lower_order_prob

    return kn_prob

# Example usage:
# file_path = 'tokenized_corpus.txt'
# corpus = read_corpus(file_path)
#%%
n = 4  # for quadgrams
quadgrams = n_gram_maker(corpus, n)
#%%
quadgram_counts, trigram_follow_counts = calculate_counts(quadgrams, n)
print("Quad complete")

#%%
trigrams = n_gram_maker(corpus, n-1)
trigram_counts, bigram_follow_counts = calculate_counts(trigrams, n-1)
print("Tri complete")

bigrams = n_gram_maker(corpus, n-2)
bigram_counts, unigram_follow_counts = calculate_counts(bigrams, n-2)
print("Bi complete")

unigrams = n_gram_maker(corpus, 1)
unigram_counts, _ = calculate_counts(unigrams, 1)
print("Uni complete")

#%%
unique_follow_counts_quad = unique_counts(quadgrams)
print("Quad unique complete")
unique_follow_counts_tri = unique_counts(trigrams)
print("Tri unique complete")
unique_follow_counts_bi = unique_counts(bigrams)
print("Bi unique complete")

#%%
quadgram_probs = kneser_ney_smoothing(quadgram_counts, unique_follow_counts_quad, trigram_follow_counts, n)
# trigram_probs = kneser_ney_smoothing(trigram_counts, unique_follow_counts_tri, bigram_follow_counts, n-1)
# bigram_probs = kneser_ney_smoothing(bigram_counts, unique_follow_counts_bi, unigram_follow_counts, n-2)
# unigram_probs = kneser_ney_smoothing(unigram_counts, {}, {}, 1)

# Displaying some probabilities for quadgrams
for quadgram, prob in list(quadgram_probs.items())[:10]:
    print(f"{quadgram}: {prob}")


# %%
def kn_smoothing(counts_dict, follow_counts_dict, lower_counts_dict, n, D=0.75):
    kn_prob=defaultdict(float)
    counts=counts_dict[n]
    follow_counts=follow_counts_dict[n]
    lower_counts=lower_counts_dict[n]


#%%
def calculate_counts_laplace(ngrams, n):
    counts = defaultdict(Counter)
    for i in tqdm(range(len(ngrams))):
        ngram=ngrams[i]
        counts[ngram[:-1]][ngram[-1]] += 1
    return counts

#%%
def laplace_smoothing(counts, vocabulary_size):
    smoothed_probs = defaultdict(lambda: defaultdict(lambda: 1/(vocabulary_size + len(counts))))
    i=0
    b=False
    for context, word_counts in counts.items():
        i+=1
        if(i%1000000==0):
            print(i, "th n-gram: ", context)
            b=True

        total_count = sum(word_counts.values())
        for word in word_counts:
            if(i%1000000==0 and b):

                print(context)
                print(word)
                print(smoothed_probs[context][word])
                b=False
            smoothed_probs[context][word] = (word_counts[word] + 1) / (total_count + vocabulary_size)
    return smoothed_probs
#%%
vocabulary = set(word for sentence in corpus for word in sentence)
vocabulary_size = len(vocabulary)

#%%
quadgram_counts=calculate_counts_laplace(quadgrams, n)
#%%
quadgram_probs_smoothed = laplace_smoothing(quadgram_counts, vocabulary_size)
#%%
for quadgram, prob in list(quadgram_probs_smoothed.items())[:10]:
    print(f"{quadgram}: {prob}")
print()
#%%
import random

def generate_text(model, initial_sequence, max_length=100):
    """
    Generate text using the given n-gram language model.

    Args:
        model (dict): The n-gram language model containing probabilities.
        initial_sequence (tuple): The initial sequence of words to start generation.
        max_length (int): Maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    current_sequence = list(initial_sequence)
    generated_text = list(initial_sequence)
    for _ in tqdm(range(max_length)):
        next_word_probs = model.get(tuple(current_sequence), {})
        if not next_word_probs:
            break  # End of sequence or unknown n-gram
        # next_word='</s>'
        # while(next_word=='</s>' ): 
        next_word = random.choices(list(next_word_probs.keys()), weights=list(next_word_probs.values()))[0]
        generated_text.append(next_word)
        current_sequence = current_sequence[1:] + [next_word]
    return ' '.join(generated_text)

#%%
# Example usage:
initial_sequence = ['<s>', '<s>', 'india']
generated_text = generate_text(quadgram_probs_smoothed, initial_sequence)
print(generated_text)
# %%
