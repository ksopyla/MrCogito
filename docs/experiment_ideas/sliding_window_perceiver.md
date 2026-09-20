Based on recent experiments E21  and some perciver critic @docs/literature_review/recurrent_memory_transformers.md @docs/literature_review/perceiver_io_latent_reasoning_critique.md 
I was thinking about the model which tries to fix some of the issues, but still uses some of perciver or cross-attention benefits. 



try to follow the 
Information theory. Tishby’s bottleneck: keep I(Z;Y) (what the answer needs), throw away the rest of I(X;Z) (arXiv:physics/0004057). A uniform mean maximises a crude “what is typical in this pack,” which is the wrong sufficient statistic for a key. Superposition: D dimensions are a packing budget, not D facts (RankMe; arXiv:2210.02885). ICAE: unique-text reconstruction is count-starved below ~4:1; 16:1 is in the failure regime for lossless rehearsal (arXiv:2307.06945). Fine-KV: synthetic recall 94% → 14% as compression goes 4× → 16× (arXiv:2412.17483). Gisting: one token is enough for an instruction gist (arXiv:2304.08467) — the opposite task. Free Perceiver C=128 over the whole sequence does not grow files with length; that is why the old 128-concept line was the wrong bandwidth story. The revisit note is explicit: C must scale with N, positionally, not “buy 16k free latents.”


Past negative observations (do not repeat that) about why it didn't work: 
1. slots where r token vectors are average - is not good idea, we want to have somthing which pics the signal from noice in each window

Positivie observation, intuitions and theory from papers, use it and improve and implement it (if possible):
1. Window latents (aggregated tokens or latents) are good and neccessary, it need to know its possitions
2. The latents shouldn't be fixes, it should be dependent of sequence length, we could decide how many tokens each latent could compress or picked by cross-attention, we should use more heads, 
3. We could create a sets of latents Zi , each latent array is fixed memory (32 vectors), those attend in a sliding window with defined stride, each Zi - attend to 8,16 times more tokens (need to be checked), eg. assuming 8x coverage and stride 192, first Z1 attend to 256 first tokens (0-255), next Z2 attends  192+256 next tokens, etc
4. Latents should have a bandwith, so this means to find a way of defining its dimensionality, my intuintion tells me that at least 4x token embeddings, but this should be aligned with window and stride, we still want a good compression ration
5 Tiny token embedding are good 128, 256 - this is intuition

Neutral, this could be ablation for future, revise with my observation (if you not agree or it is cheep to implement), by default do not implement that, keep it simple
1. TinyHashed Embeddings - I didn't see much differen
2. SWA (sliding-window attention) as a mixing nearby tokens - need to be proved, I dont know if it helps or not 

