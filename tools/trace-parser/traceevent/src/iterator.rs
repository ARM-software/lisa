use core::{
    cmp::{Ordering, Reverse},
    fmt::{Debug, Error, Formatter},
    iter::Iterator,
    ops::{Deref, DerefMut},
};
use std::collections::BinaryHeap;

pub struct SavedIterator<I>
where
    I: Iterator,
{
    iter: I,
    item: Option<I::Item>,
}

impl<I> Debug for SavedIterator<I>
where
    I: std::iter::Iterator,
    <I as Iterator>::Item: Debug,
{
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_struct("SavedIterator")
            .field("item", &self.item)
            .finish_non_exhaustive()
    }
}

impl<I> PartialEq for SavedIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.item == other.item
    }
}

impl<I> Eq for SavedIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: PartialEq,
{
}

impl<I> PartialOrd for SavedIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.item, &other.item)
    }
}

impl<I> Ord for SavedIterator<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&self.item, &other.item)
    }
}

pub struct MergedIterator<I>
where
    I: IntoIterator,
{
    curr: Reverse<SavedIterator<I::IntoIter>>,
    heap: BinaryHeap<Reverse<SavedIterator<I::IntoIter>>>,
}

impl<I> MergedIterator<I>
where
    I: IntoIterator,
    SavedIterator<<I as IntoIterator>::IntoIter>: Ord,
{
    pub fn new<II: IntoIterator<Item = I>>(iterators: II) -> Option<Self> {
        let mut iterators = iterators.into_iter().map(|i| i.into_iter());

        let first = iterators.next()?;
        let curr = Reverse(SavedIterator {
            item: None,
            iter: first,
        });

        let mut heap = BinaryHeap::with_capacity(iterators.size_hint().0);
        heap.extend(iterators.filter_map(|mut iter| match iter.next() {
            None => None,
            item => Some(Reverse(SavedIterator { item, iter })),
        }));

        Some(MergedIterator { curr, heap })
    }
}

impl<I> Iterator for MergedIterator<I>
where
    I: IntoIterator,
    SavedIterator<<I as IntoIterator>::IntoIter>: Ord,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.curr.0.iter.next() {
            // We exhausted the last selected iterator, so we just drop it and
            // select the highest heap item.
            None => {
                self.curr = self.heap.pop()?;
            }
            new => {
                self.curr.0.item = new;
                // If the highest in the heap is higher than the last selected
                // iterator, we select a new iterator and put back the other one in
                // the heap.
                if let Some(mut highest) = self.heap.peek_mut() {
                    if highest.deref() > &self.curr {
                        core::mem::swap(highest.deref_mut(), &mut self.curr);
                    }
                }
            }
        }

        // SAFETY: We guarantee that we will not call next() on the iterator
        // that produced this value until the next iteration.
        core::mem::take(&mut self.curr.0.item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test(iterators: Vec<Vec<u32>>, expected: Vec<u32>) {
        // eprintln!("=============================");
        // eprintln!("{:?} => {:?}", iterators, expected);
        let merged = MergedIterator::new(iterators).expect("No iterator to merge");
        assert_eq!(merged.collect::<Vec<_>>(), expected);
    }

    #[test]
    fn iterator_test() {
        test(vec![vec![1, 3, 5], vec![2, 4]], vec![1, 2, 3, 4, 5]);
        test(vec![vec![1, 3], vec![2, 4, 6]], vec![1, 2, 3, 4, 6]);

        test(vec![vec![1, 3, 5], vec![2, 4, 6]], vec![1, 2, 3, 4, 5, 6]);
        test(vec![vec![2, 4, 6], vec![1, 3, 5]], vec![1, 2, 3, 4, 5, 6]);
        test(vec![vec![], vec![2, 4, 6]], vec![2, 4, 6]);
        test(vec![vec![1, 3, 5], vec![]], vec![1, 3, 5]);

        test(
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        test(
            vec![vec![7, 8, 9], vec![4, 5, 6], vec![1, 2, 3]],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        test(
            vec![vec![], vec![4, 5, 6], vec![1, 2, 3]],
            vec![1, 2, 3, 4, 5, 6],
        );
        test(
            vec![vec![4, 5, 6], vec![], vec![1, 2, 3]],
            vec![1, 2, 3, 4, 5, 6],
        );
        test(
            vec![vec![4, 6], vec![5], vec![1, 2, 3]],
            vec![1, 2, 3, 4, 5, 6],
        );
        test(vec![vec![4, 6], vec![], vec![1, 2, 3]], vec![1, 2, 3, 4, 6]);

        test(vec![vec![4, 6], vec![1, 2, 3], vec![]], vec![1, 2, 3, 4, 6]);
        test(vec![vec![], vec![4, 6], vec![1, 2, 3]], vec![1, 2, 3, 4, 6]);
    }
}
