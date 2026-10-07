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
