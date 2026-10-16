# E35 — Latent messages between agents (memory · writing · sending, kept apart)

- **Status:** started 2026-10-16. The author asked for an agent-to-agent system built from E33, trained small on
  Odra and Polonez, and then demonstrated: "a clear separation between memory, writing to the memory, and sending
  to another agent"; compression "according to the article Cramming 1568 tokens…", with no quantization; "work
  until you reach the goal".
- **Serves:** the Vision's last step: agents exchanging latent concepts instead of text. Long input (the goal is
  1M–10M tokens) goes into a memory on the sender's side. A few real-valued vectors go to the receiver, which
  decodes answers.
- **Builds on:**
  - E31c / E33: the page writer and the notebook;
  - E33e: order among matching notes;
  - the text checks: data, exams, and the dense reference;
  - Cramming (arXiv 2502.13063): message vectors enter the receiver as input embeddings, read by every layer.
- **Owner / dates:** ksopyla · opened 2026-10-16

## The bet
A sender that writes a long document into a page-parallel latent memory can transmit **K = 16 vectors**, read by
a separate receiver as leading input embeddings. That is enough for the receiver to answer fact questions about
documents up to 4k tokens, at least 4× above guessing, with no access to the document itself. The memory side
then scales to far longer documents at the same message size.
- Compression is counted as Cramming counts it: document tokens per transmitted vector, 256 at 4k.
- Bits are reported too. Each vector is 576 numbers at 16 bits, about 9.2k bits.

Why it could work:
- **Capacity.** A frozen 160M language model decodes 80 tokens from one optimized vector, and an 8B model
  decodes 1,568. Capacity tracks the text's surprise under the receiver, not its length, and grows almost
  linearly with the number of vectors.
- **What has to cross.** A text-checks document of any length holds about 12 facts (~500 bits). The filler
  stories are predictable or irrelevant. So the message must carry the facts, not the text: distillation, not
  transcription.

## The four parts, each with one job (the separation the author asked for)

```
AGENT A (sender)                                                  AGENT B (receiver)
text ──► WRITER ──► MEMORY ──► TRANSMITTER ══ message: K × 576 ══► RECEIVER: [message | own text] → answer
         pages of 256      entries: latent (512),        K learned queries read the memory
         (stride 192),     source position, page;        (cross-attention + order among matches),
         2 page layers,    append-only, page-parallel    self-attention among the K slots,
         32 latents/page   → linear in length            output scaled like input embeddings
```

- **Writer** (`LatentMemoryWriter`, from E31c run 2's trained weights):
  - its own token embedding, separate from the receiver's;
  - pages are written independently, so 1M tokens is about 5,200 pages written in parallel.
- **Memory** (`LatentMemory`): the written entries, their source positions, and a valid mask. Nothing reads raw text after writing.
- **Transmitter** (`Transmitter`, new):
  - K learned query slots, 2 blocks;
  - each block has cross-attention over all memory entries, self-attention among slots, and a feed-forward layer;
  - a learned "nothing" entry, so an empty or irrelevant memory reads as nothing;
  - order among matching entries: the E33e count, per-head strength starting at 0;
  - the output passes through a norm initialised like the receiver's input embedding.
- **Receiver** (dense decoder, from the best dense run on the bridge data): reads the K vectors as the first K
  input positions. Every layer attends to them, as in Cramming. Its own text follows: questions here; "additional
  transmitted data" in the demo.
- **The only thing crossing between the agents is the message tensor.** In code these are separate calls:
  - `write(text) → memory`;
  - `send(memory) → message`;
  - `answer(message, text) → tokens`.

## Training (first round)
- **Data:** text-checks world documents, 8 questions per document.
  - The sender gets the document body only.
  - The receiver gets the questions and answers.
  - Story rows (45 % of tokens) train the receiver alone (empty memory), to keep its language.
  - Rows: `BOS document EOS BOS questions EOS`; the model splits at the first EOS that is followed by BOS.
- **Loss:** next-token loss on the receiver's tokens only. Gradients reach the transmitter and the writer
  through the message.
- **Init:** writer and sender embedding from E31c run 2, receiver from dense on the bridge data, transmitter fresh.
- **Budget:** 0.5B tokens. Same optimizer recipe as the text checks (AdamW, 5e-4 for warm-started weights, cosine).
- **Variants**, one change each:
  - K = 16 (main);
  - K = 4 (4× more compression);
  - K = 64;
  - order term off.

## Measures
- **A2A exam:** the text-checks exam items split into sender (document) and receiver (question). Exact answer
  per task at 1k / 4k / 16k, then longer documents (64k, 256k, 1M) for lookup.
- **Controls:**
  - receiver with an empty message (the floor: what the receiver guesses);
  - dense reading the whole document (the ceiling at ≤ 4k);
  - E31c reading its full notebook (the same memory, no compression into K vectors).
- **Cramming probe** (channel capacity at our scale): per-sample optimization of 1–4 input vectors for the frozen
  receiver, measuring the most tokens reconstructed exactly. This is the ceiling the trained transmitter is compared against.

## Success / kill
- **Pass (round 1):**
  - lookup, keyed and quote at 4k ≥ 40 % exact (guessing 6–8 %; dense reading the text gets 80–99 %);
  - the empty-message control at guessing;
  - story loss of the receiver within 0.03 of its start.
- **Stretch:** lookup at 16k ≥ 40 % with the same 16 vectors (4× the training length).
- **Kill this form:** all fact tasks below 15 % at 4k with K = 64 after 0.5B tokens. The transmitter cannot read
  the memory, so the read needs a request from the receiver, not a broadcast.

## Not in scope (round 2+)
- request-conditioned sending (the receiver's question steers what is sent);
- a reconstruction objective (Cramming-style autoencoding) mixed in;
- test-time refinement of the message by gradient steps against the receiver;
- different architectures for sender and receiver.

## Result
_(to fill)_
