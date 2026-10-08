# Shared THC host-data contract

`green/integrals/thc_factor_data.h` owns descriptor validation, factor storage,
and original-Q pair reconstruction. It contains no solver or CUDA scheduling.
CPU and GPU consumers must use the coordinated THC feature commits.

The opt-in flags are `interaction_representation=df|thc`, explicit
`thc_mode=reconstruct|native`, and `thc_factor_memory_mb` (default 512).
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
