/* SPDX-License-Identifier: Apache-2.0 */
pub const fn concat<const N: usize>(slices: &[&'static str]) -> [u8; N] {
    let mut out: [u8; N] = [0; N];
    let mut out_i: usize = 0;

    let nr_slices = slices.len();
    let mut slice_i = 0;
    while slice_i < nr_slices {
        let slice = slices[slice_i];
        slice_i += 1;

        let slice: &[u8] = slice.as_bytes();

        let slice_len = slice.len();
        let mut curr_slice_i: usize = 0;

        while curr_slice_i < slice_len {
            out[out_i] = slice[curr_slice_i];
            out_i += 1;
            curr_slice_i += 1;
        }
    }
    out
}
