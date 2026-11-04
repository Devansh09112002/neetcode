import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        # STEP 1: Q, K, V GENERATE KARNE KI LAYERS

        # embedding_dim = input token embedding ka size
        # attention_dim = Q, K, V vectors ka size
        # Example: embedding_dim=16, attention_dim=8

        # Key: Token kis information ko represent karta hai?
        # Input 16 numbers -> Output 8 numbers
        self.key_gen = nn.Linear(
            embedding_dim, attention_dim, bias=False
        )

        # Query: Token kis information ko search kar raha hai?
        # Input 16 numbers -> Output 8 numbers
        self.query_gen = nn.Linear(
            embedding_dim, attention_dim, bias=False
        )

        # Value: Token ki information jo dusre tokens ko milegi
        # Input 16 numbers -> Output 8 numbers
        self.value_gen = nn.Linear(
            embedding_dim, attention_dim, bias=False
        )

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:

        # STEP 2: INPUT EMBEDDINGS SE Q, K, V BANAO

        # Maan lo 4 tokens hain, har token ka embedding size 16 hai
        # Input shape = (1, 4, 16)
        # 1 = batch size, 4 = tokens, 16 = embedding dimension

        # Har token ka Key vector banao
        k = self.key_gen(embedded)      # (1, 4, 8)

        # Har token ka Query vector banao
        q = self.query_gen(embedded)    # (1, 4, 8)

        # Har token ka Value vector banao
        v = self.value_gen(embedded)    # (1, 4, 8)


        # STEP 3: ATTENTION SCORES CALCULATE KARO

        # Q aur K ka dot product similarity score deta hai
        # High score -> token ko zyada attention mil sakti hai
        #
        # k shape = (1, 4, 8)
        # transpose(k) -> (1, 8, 4)
        # q @ k.T -> (1, 4, 4)
        #
        # Har token ka har token ke saath similarity score
        scores = q @ torch.transpose(k, 1, 2)

        # Sequence length aur attention dimension nikalo
        context_length, attention_dim = k.shape[1], k.shape[2]

        # Scores ko sqrt(attention_dim) se divide karo
        # Example: attention_dim=8 -> sqrt(8)=2.828
        #
        # WHY? Dot products large ho sakte hain
        # Scaling softmax ko excessively sharp hone se bachati hai
        scores = scores / (attention_dim ** 0.5)


        # STEP 4: CAUSAL MASKING

        # GPT-style autoregressive model mein token future nahi dekh sakta
        #
        # Example: "I love AI"
        # I    -> sirf I ko dekh sakta hai
        # love -> I aur love ko dekh sakta hai
        # AI   -> I, love, AI ko dekh sakta hai

        # Lower triangular matrix banao
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        lower_triangular = torch.tril(
            torch.ones(context_length, context_length)
        )

        # Jahan 0 hai wahan future tokens blocked hain
        mask = lower_triangular == 0

        # Future token scores ko -infinity kar do
        # Softmax ke baad inki probability 0 ho jayegi
        scores = scores.masked_fill(mask, float("-inf"))


        # STEP 5: SOFTMAX

        # Scores ko attention weights mein convert karo
        # Example: [2.0, 1.0] -> [0.731, 0.269]
        #
        # dim=2 -> har query token ke liye keys par softmax
        # Har row ke attention weights ka sum = 1
        scores = nn.functional.softmax(scores, dim=2)


        # STEP 6: WEIGHTED SUM OF VALUES

        # Attention weights ko Value vectors se multiply karo
        #
        # Example: token A ka attention:
        # 70% token A + 30% token B ki information
        #
        # scores shape = (1, 4, 4)
        # v shape      = (1, 4, 8)
        # output       = (1, 4, 8)
        #
        # Har token ko context-aware representation milegi
        return torch.round(scores @ v, decimals=4)