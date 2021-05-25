import torch
import torchtext.vocab as vocab

class RNNDecoder(torch.nn.Module):
    def __init__(self, input_size=40000, hidden_size=50, n_layers=1, cache_dir='/home/qlwang/img_cap/res/glove', name='6B'):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # 词向量输入层
        self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        glove = vocab.GloVe(name=name, dim=hidden_size, cache=cache_dir)
        self.word2dict = glove.stoi
        self.word2dict = {k:v for k,v in self.word2dict.items() if v<input_size}
        self.word2index = glove.itos[:input_size]
        self.word2vec = glove.vectors[:input_size]
        del glove

        self.embedding.weight.data.copy_(self.word2vec)
        self.embedding.requires_grad=False

        # RNN-GRU
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers)

        # 词向量输出层

    
    def getWordByIndex(self, index):
        return self.word2index[index]
    
    def getVectorByName(self, word):
        return self.word2vec[self.word2dict[word]]

    def knn(self, w, k):
        cos = torch.matmul(self.word2vec, w) / ((torch.sum(self.word2vec * self.word2vec, dim=1) + 1e-9).sqrt() * torch.sum(w * w).sqrt())
        _, topk = torch.topk(cos, k=k)
        return topk      

    def getSimilarTokens(self, word, num_tokens):
        w = self.getVectorByName(word)
        topk = self.knn(w, num_tokens+1)[1:]
        return [self.getWordByIndex(i) for i in topk]


if __name__ == '__main__':
    rnn = RNNDecoder()
    print(rnn.getVectorByName('sausages'))
    print(rnn.getSimilarTokens('shit', 3)) 

