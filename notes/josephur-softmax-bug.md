Josephur [COS],  — 3:24 PM
Well they arent all done yet, most of them live in a text file I'm still working on the engine basics 😄
Did you come across a bug in Intels v_exp_f32 ?
I'm inclined to trust Claude on this but
 `v_exp_f32` is the Gaudi 2 TPC's built-in instruction for e^x on 32-bit floats. Every attention softmax uses it, ours included until today, because softmax is e^(score − max).

**The bug:** at **exactly 42 input values** between −104 and 0, it returns about **twice the right answer**.

- The worst case for softmax is x = −0.693147182, the float closest to −ln 2. e^x should be **0.5**, but `v_exp_f32` returns **1.0**.
- Everywhere else it's fine, within 2 ulp.
- The correctness agent proved this exhaustively: it ran **all 1.12 billion floats in [−104, 0] through the card**, and the card and the simulator agree bit for bit. The full list of bad inputs is in `research/exp_bug/vendor_bad_f32.txt`. That's tier A, measured evidence.

**Why it happens**, from the disassembly, so that part is tier C, inferred. The usual way to compute e^x is to split x into a whole number of powers of 2 plus a small remainder: 
 1. It computes `n = rint(x · log₂e + 0.5)`.
2. For inputs where x · log₂e is just below an **odd** whole number k, the 32-bit sum rounds to exactly k + ½. "Round half to even" then picks k + 1 instead of k, so the split is off by one.
3. That leaves a remainder just below −1, outside the range the lookup table expects. The table's correction step only fixes the *upper* edge of its range, so the value is read one power of two too high: a factor of 2.

That's why there's exactly one bad input per odd k, which gives the 42.

**Does it hit real inference?** Yes, rarely. With the vendor exp, decode attention on the 4 test prompts produced **25 wrong attention weights out of 122 million**, about 1 in 4.9 million. Each one counted a single token twice as heavily as it should have, in one layer for one step. That's far too rare and too small to visibly change output here, but it's genuinely wrong math, and with enough generation it could tip a close token choice.

**The fix:** every decode softmax and combine kernel now uses **EXPD**, our own 13-operation exp (a Cody–Waite split plus a degree-6 polynomial). It's the same formula our prefill already used:
 
- **Accurate:** within 1 ulp everywhere in [−87, 0], with no bad inputs. Rechecked on 121 million real decode scores: 0 wrong.
- **Faster:** 16–34% fewer TPC cycles in those kernels, and slightly faster steps (K=4: 32.32 → 32.28 ms).
- **Same tokens:** identical on the test prompts, except three forks that are all documented BF16 near-ties (margins of 0 to 0.125).
- **Reversible:** a build flag (`-DVENDOR_EXP`) restores the vendor exp. 
