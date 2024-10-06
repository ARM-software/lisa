
This folder contains rustc target JSON definitions. They are equivalent to a
custom --target triplet passed to rustc.

In order to establish one, the easiest way is start from an existing target
that is close enough (typically a bare-metal target) and adapt it for kernel
use, e.g.:

$ rustc +nightly -Z unstable-options --print target-spec-json --target=$TRIPLET

Note that these JSON targets are barebone on purpose to be easily maintainable.
Any architecture-agnostic options such as -Cno-redzone=y or the
relocation-model are passed as RUSTFLAGS in the Makefile.
