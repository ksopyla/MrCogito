
# Embedding spaces encoding capabilites 

I have tried to understand the information encoding capabilities of the embedding spaces via the Shannon and other information theory concepts.

What I wanted to understand better is the relationship between the number of possible values for the variable K and the dimension of the embedding space D.


## Information encoding capabilities

Lets define main two variables: 

* K - number of possible values for the variable, eg for a binary K=2, for a ternary K=3, for a decimal K=10
* D - dimension of the embedding space, eg for a 1D embedding space D=1, for a 2D embedding space D=2, for a 3D embedding space D=3

Then the number of possible values for the variable is $K^D$.

The maximum value encoded by the embedding space is $K^D - 1$.

What I wanted to understand better is the relationship between the number of possible values for the variable K and the dimension of the embedding space D.

To increase the information encoding capabilities of the embedding vector we can:
1. Increase the number of possible values for the variable K
2. Increase the dimension of the embedding space D

I want to find an answer to:
1. If I will increase the D by 1, how many D-dim with K values vecors I can fit into the same embedding space?
    - the relationship is: $K^{D+1} / K^D = K$
2. How this relationship apply to computer floating point precision, fp16 and bfloat16?
3. the above assumes that the whole possible ranges of values in learned embedding space are used, are there any reserach that suggests that only a subset of the possible values are used?
     1. How training algorithms influence the usage of the possible values?
     2. Are embedding vectors are uniformly distributed or not? Is this dependent on the training algorithm or other factors?
4. Could we futher compress the learned embedding vectors, by some postprocessing step, like quantization, clustering, etc.?
5. Based on the about questions, and my ConceptEncoder ideas @docs/research-notes/concept_encoder_notes.md, what is the optimal dimension for Concepts and tokens embeddings? If concepts should capture and compress the information from the tokens, what is the optimal number of concepts?





