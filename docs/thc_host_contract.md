# Shared THC host-data contract

`green/integrals/thc_factor_data.h` owns descriptor validation, factor storage,
and original-Q pair reconstruction. It contains no solver or CUDA scheduling.
CPU and GPU consumers must use the coordinated THC feature commits.

The opt-in flags are `interaction_representation=df|thc`, explicit
`thc_mode=reconstruct|native`, and `thc_factor_memory_mb` (default 512).
The registered `thc_cuda_aux_gemm3m` flag defaults to false and is consumed only
by native GPU GW auxiliary contractions; it does not change loader behavior.
The helper validates construction, accepted state, retained Q, complete schema,
input dimensions/fingerprint, k order, q transfers, source maps, set identity,
all factor shapes/dtypes/checksums/finite values, and correction sidecars.
It retains X plus one cached q core, or X plus all cores when requested.
The memory budget covers resident factors; returned M copies, reconstruction
tiles and solver workspaces are additional. Factor handles are immutable,
set-specific; the lazy q cache is mutex-protected. Reconstruction tiles I,
original Q and orbital rows, and returns synchronous owned data.

Canonical ordering:
`L_Qmn=sum_I conj(X_ki_Im)*X_kj_In*M_q_IQ`, `q=kj-ki` modulo reciprocal
lattice. `source_representative` is reserved for legacy raw-buffer correction
consumers; ordinary consumers request the full oriented canonical pair.
Existing AO/auxiliary symmetry operations are not interpolation operations.
The helper lives beside existing symmetry dependencies for common ownership,
but it does not alter the existing symmetry implementation.

The separate `green/tensors/thc_gw_fft.h` implements a full regular-mesh
embedding, complex host FFT correlations, and hardware-neutral native GW
algebra. Consumers own MPI scheduling; the GPU consumer uses the same mesh maps
with a separate resident CUDA/cuFFT implementation.
It is separate from the factor loader and does not reconstruct V. It requires
the ndarray/Eigen/grids interfaces already present in MBPT and GPU consumers.
The k mesh may be shifted; its integer coset embedding and actual Bloch values
are retained. An explicit conservative all-q workspace check precedes allocation.

CPU transforms use the opaque `thc_host_fft` backend with cached batched FFTW
plans when headers/library are found. `GREEN_THC_USE_FFTW=OFF` forces Eigen
fallback; FFTW is optional, and its headers stay out of public CUDA headers.
Plan construction/destruction is serialized; each solver owns its executor.

Screening supports point space or the exact original-Q identity
`P=M^H chi M; (I-P) C=P; Wc=M C M^H`. Auto uses auxiliary space when Q < I.
The loader and archive retain the same M, Q and validation contract; selecting
the smaller solve introduces no approximation or additional truncation.

Auxiliary GW stores Q-by-Q tau/frequency histories. The time-independent M
commutes with the linear IR transforms, so compression precedes the forward
transform and Wc expansion follows the inverse transform at the current tau.
The summed-spin bubble is compressed only in the mirrored half of tau;
Hermitian symmetrization commutes with M^H(.)M and the other half is copied.
Current-tau Wc is shared across spins. Momentum FFT fields remain point-sized:
q-dependent M cannot commute with that transform. Workspace estimates include
the compact histories and the remaining point projection/momentum/Sigma fields.


## Sigma contraction and execution controls

The `feature/thc-sigma-optimization` branch adds exact complex-double execution
alternatives for native scalar, full-BZ GW. DF remains the default interaction
representation. These controls do not change the fitted factors or fit tolerances.

- `--thc_gw_sigma point|orbital|auto` defaults to `point`. `orbital` requires
  auxiliary screening and direct momentum sums. It constructs fixed vertices
  `V[k,kp,A,a,i] = sum_p conj(X[k,p,a])*X[kp,p,i]*M[q,p,A]`, then contracts
  `Sigma[k] -= sum_(kp,A,B) V_A G[kp] V_B^H C[q,A,B]/Nk`. The screened core `C`
  stays in Q space; Sigma never expands it into a point matrix. The CPU and GPU
  layouts differ internally and are tested against independent point expressions.
  The cache scales as `Nk*Nq*n*n*Q` complex values and can be reused across solver
  iterations. Cold-run timings include its construction. `auto` uses a conservative
  cached-point operation estimate with a 10% margin, not a timing guarantee.
  In unsupported momentum/screening modes it selects `point`. GPU auto also
  selects point when vertex workspace does not fit or a tau batch is requested.
- `--thc_fft_reuse_screening true|false` defaults to `true`. In momentum FFT mode
  it transforms the screened interaction once per tau and shares it across spins.
  The original non-conjugating correlation, negative-frequency mapping and shifted
  mesh values are preserved. Two-spin Sigma uses five FFT calls per tau instead
  of six. The complete Nt=110 example uses 880 rather than 990 calls, including
  the unchanged bubble stage. A one-point mesh needs no transforms.
- `--thc_cpu_threads N` defaults to 1, allows 1..64, and bounds parallel Sigma
  tau workers by the declared workspace. It applies to auxiliary/direct CPU GW
  with retained owned-q histories. Streaming falls back to one worker. Each worker
  writes disjoint tau slices; q accumulation order is preserved. Use one BLAS
  thread when testing these workers to avoid nested CPU oversubscription.
- `--thc_cuda_sigma_batch N` defaults to 0. Values 1..32 batch point-space
  projection/backprojection over spins and up to N adjacent tau slices. A final
  partial tile is supported. Extra projected/point fields consume workspace;
  this option is incompatible with explicit orbital Sigma. Zero preserves the
  original per-spin point path. Benchmark the size for the intended dimensions.

MBPT enables OpenMP when available with `GREEN_THC_OPENMP=ON` (default). Set it
OFF for a serial build; requesting CPU workers above one then produces an error.
A pre-existing GF2 tau loop was rewritten with one induction variable to make it
valid OpenMP syntax, preserving all original tau indices. Rebuild the coordinated
symmetry, GPU and MBPT revisions together after changing resident APIs.
