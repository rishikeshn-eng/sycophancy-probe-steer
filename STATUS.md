# STATUS — sycophancy-probe-steer

- Repo: https://github.com/rishikeshn-eng/sycophancy-probe-steer (public, `main`)
- Site: https://rishikeshn-eng.github.io/sycophancy-probe-steer/ (GitHub Pages from `main:/docs`)

## Headline result (Qwen2.5-1.5B-Instruct, held-out test split)
- The model barely discriminates good from bad pushback. It switches to a user-suggested wrong answer 88.7% of the time when it was right, and to a user-suggested correct answer 91.9% of the time when it was wrong. 0.5B switches about 100% of the time either way.
- "Will cave" is linearly decodable, but surface metadata alone gets AUROC 0.90. With wording held fixed, the signal emerges at layers ~16–23 (within-strata AUROC 0.89–0.92 at the decision token, 0.72–0.76 before the reply starts).
- The best CAA steering vector (pushback, layer 22, α = −0.5) cuts capitulation from 89.5% to 79.2%, but acceptance of correct corrections falls from 94.0% to 82.8%. Discrimination is unchanged, and free-form TriviaQA accuracy drops from 55% to 36%. A random direction of equal norm reduces plain "are you sure?" flips as much as the CAA vectors do.

## Cuts / not done (also in the README Limitations)
- Time-boxed sweep: 1.5B covers 5 layers × 2–3 α; 0.5B covers 7 layers. Final test sets have 300–400 items per metric; MMLU has 300.
- Gemini coherence judging is incomplete because the shared API key ran out of quota (`gemini-2.5-flash` is no longer served). The local repeated-trigram rate is reported for every config.
- Two GPU jobs hung in Metal (macOS MPS) near the deadline and were killed: the 0.5B held-out steering run (0.5B steering is reported from its selection-split sweep only) and a layer-22 random-direction control (the only random control is at layer 6).
- Gemini coherence: only 14/90 items judged (all unsteered, all 5/5) before the shared key's quota ran out.
- `meg-tong` `feedback` set not evaluated. No Hugging Face upload (no token).
