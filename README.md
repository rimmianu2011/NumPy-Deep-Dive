# NumPy Lab — FAANG-Level Hands-On

**Goal:** Build deep intuition for NumPy internals, vectorization, and performance — the way FAANG expects ML engineers to think.

**Outcome:** Students can write fast, memory-efficient, interview-ready NumPy code and explain *why* it is efficient.

---

## Lab Rules (FAANG Style)

- ❌ No Python loops unless explicitly allowed
- ✅ Prefer vectorization & broadcasting
- ✅ Always analyze time & space complexity
- ✅ Explain *why* NumPy code is faster

---

## Section 1 — ndarray Fundamentals

### Task 1.1: Array Creation & Shapes

Create the following **without loops**:

- A 1D array of numbers from 0 to 99
- A 2D array of shape `(10, 10)`
- A 3D array of shape `(4, 5, 3)`

**Checkpoint Questions:**

- What does `ndarray.shape` represent?
- Why does NumPy store data in contiguous memory?

---

### Task 1.2: dtype & Memory

1. Create the same array using `int64` and `float64`
2. Compare memory usage using `.nbytes`

**Interview Angle:**

- Why does dtype matter in large ML pipelines?

---

## Section 2 — Indexing, Views & Copies

### Task 2.1: Fancy vs Basic Indexing

- Slice a 2D array to extract every alternate row
- Modify the slice and observe changes in the original array

**Concept Check:**

- Views vs copies
- When does NumPy allocate new memory?

---

### Task 2.2: Boolean Masking

Given an array of random values:

- Extract all values `> mean`
- Replace negative values with `0`

**Interview Angle:**

- Why boolean masking is preferred over loops

---

## Section 3 — Broadcasting (Critical for FAANG)

### Task 3.1: Broadcasting Rules

Given:

- `A` of shape `(1000, 50)`
- `b` of shape `(50,)`

Tasks:

1. Add `b` to every row of `A`
2. Normalize each row of `A`

**Explain:**

- Broadcasting rules step-by-step

---

### Task 3.2: Manual Broadcasting Trap

- Try to subtract a column vector incorrectly
- Fix the shape issue

**Interview Gotcha:**

- Common broadcasting mistakes candidates make

---

## Section 4 — Vectorization vs Loops

### Task 4.1: Loop → Vectorized

Given an array `X` of size `1,000,000`:

- Compute `(X − mean) / std`
- First using a Python loop
- Then using NumPy vectorization

Measure execution time.

**Discussion:**

- Why vectorization is faster
- Role of C-level execution

---

### Task 4.2: Pairwise Distance (Classic FAANG)

Given matrix `X` of shape `(n, d)`:

- Compute pairwise Euclidean distances **without loops**

**Hint:** Use broadcasting + algebra

---

## Section 5 — Numerical Stability

### Task 5.1: Softmax Instability

1. Implement naive softmax
2. Observe overflow
3. Fix using numerical stability trick

**Interview Question:**

- Why subtracting max works?

---

## Section 6 — Linear Algebra with NumPy

### Task 6.1: Dot, MatMul & Shapes

- Perform matrix multiplication with valid & invalid shapes
- Compare `np.dot`, `@`, and `np.matmul`

---

### Task 6.2: Solving Linear Systems

- Solve `Ax = b` using NumPy
- Verify the solution

**ML Context:** Linear regression closed-form

---

## Section 7 — Performance & Memory Tricks

### Task 7.1: In-Place Operations

- Compare in-place vs out-of-place operations
- Measure memory usage

---

### Task 7.2: Strides (Advanced)

- Inspect `.strides` of arrays
- Explain what they mean

**FAANG Bonus:** Why strides matter for performance

---

## Section 8 — Mini Case Study (ML-Oriented)

Given a dataset matrix `X` `(10,000 × 100)`:

- Normalize features
- Compute covariance matrix
- Extract top-k principal components (**no sklearn**)

---

## Submission Expectations

Students must submit:

- Clean NumPy notebook
- Markdown explanations for each section
- Time & space complexity notes

---

## FAANG Interview Evaluation Rubric

| Skill                  | Evaluated |
|------------------------|-----------|
| Vectorization          | ✅        |
| Broadcasting intuition | ✅        |
| Numerical stability    | ✅        |
| Memory awareness       | ✅        |
| Explanation clarity    | ✅        |

---

## Stretch Problems (Optional)

- Implement cosine similarity matrix
- Optimize pairwise distance further
- Compare NumPy vs pure Python memory usage
