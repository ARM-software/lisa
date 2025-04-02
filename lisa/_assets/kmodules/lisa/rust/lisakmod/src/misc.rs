/* SPDX-License-Identifier: GPL-2.0 */

// Join the parameters using the given separator, avoiding any trailing separator
macro_rules! join{
    ($sep:expr, $first:expr $(, $rest:expr)* $(,)?) => {
        ::core::concat!($first $(, $sep, $rest)*)
    };
}
pub(crate) use join;
