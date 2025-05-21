# Trellis Quantization (TCQ) 

Trellis-Coded Quantization (TCQ) is a structured quantization technique that compresses a sequence of values by finding an optimal path through a constrained state machine known as a trellis. Unlike scalar quantization, which treats each value independently, TCQ jointly optimizes the quantization of a sequence to minimize total distortion.

This method is especially effective in compressing neural network weight matrices, where block-level patterns can be exploited for better accuracy and compression.

---

## 1. Motivation

Traditional quantization, such as uniform scalar quantization, minimizes local error by rounding each value independently. However, this can lead to suboptimal results when the data has structure or correlation, as in neural network weights.

TCQ addresses this by encoding an entire block or sequence of values together, enforcing path constraints via a trellis structure. The global sequence-level optimization leads to significantly lower average distortion.

---

## 2. The Structure of a Trellis

A trellis consists of:

- **States**: Each representing a configuration at a timestep
- **Transitions**: Connections between states allowed by a fixed rule
- **Output values**: Each transition emits a quantized value

Each input value is not assigned a quantization index independently. Instead, the entire sequence is represented as a path in the trellis, and the quantized values are determined by that path.

---

## 3. Quantization as a Path-Finding Problem

Given:
- A quantization codebook: e.g., `C = [-1.0, -0.5, 0.0, 0.5, 1.0]`
- A sequence of input values: e.g., `[0.4, -0.7, 0.1, 0.3]`
- A state transition rule: e.g., each state can move to 2 of the next layer’s states

Goal: Find the path through the trellis that results in quantized values minimizing the total squared error to the input.

---

## 4. Dynamic Programming (Viterbi Algorithm)

The Viterbi algorithm is used to efficiently find the minimum-error path:

1. At each time step `t`, for each possible state `s`:
   - Compute the minimum cumulative cost to reach that state from the previous layer
   - Store the corresponding backpointer

2. At the final time step, select the state with the lowest total cost

3. Backtrack from the final state to reconstruct the optimal path

---

## 5. Example

Suppose:

- Codebook: `[-1, 0, 1]`
- Input sequence: `[0.2, -0.4, 0.5]`
- Trellis: Each state has 2 allowed transitions to the next layer

Rather than:
Scalar Quantization → [0, -0, 1]
Total Error ≈ (0.2-0)^2 + (-0.4-0)^2 + (0.5-1)^2 ≈ 0.04 + 0.16 + 0.25 = 0.45


TCQ would try sequences like:
Path A: [0, -1, 1] → Error = (0.2-0)^2 + (-0.4+1)^2 + (0.5-1)^2 = 0.04 + 0.36 + 0.25 = 0.65
Path B: [1, 0, 0] → Error = (0.2-1)^2 + (-0.4-0)^2 + (0.5-0)^2 = 0.64 + 0.16 + 0.25 = 1.05
Path C: [0, 0, 1] → Error = 0.04 + 0.16 + 0.25 = 0.45 ✔ (same as scalar)


Then Viterbi will choose the best path with valid state transitions and minimal error.

---

## 6. Encoding: How TCQ Paths Are Represented

Each path is encoded as a binary stream representing:

- The initial state
- A sequence of branch indices (e.g., which of the K possible branches was taken at each step)

This allows compression of N quantized values into fewer bits than scalar quantization.

Example: With K=2 branches and N=8 values, only log2(K^N) = N bits are needed.

---

## 7. Bitshift Trellis in QTIP

QTIP uses a specialized version of TCQ called the **Bitshift Trellis**, where:

- Each state is represented by an L-bit integer
- Each step:
  1. Shifts the current state left by 1 bit
  2. Appends a new input bit
  3. Masks to retain only L bits

This efficient structure allows for extremely fast computation and compact code representation.

```text
Example (L = 12):

Current state: 011011000001
Input bit: 1
New state: (011011000001 << 1) | 1 = 110110000011