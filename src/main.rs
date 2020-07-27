#![allow(dead_code)]
#![allow(unused_imports)]
#![feature(drain_filter)]
#![feature(trait_alias)]
use std::iter::{once, repeat};
#[derive(Debug, Clone, PartialEq)]
enum Entry {
    Possible(Vec<Entry>),
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
}
impl Default for Entry {
    fn default() -> Self {
        Entry::Possible(vec![])
    }
}
trait CompleteIterator<S> = DoubleEndedIterator<Item = S> + ExactSizeIterator<Item = S>;
trait Row<S> = DoubleEndedIterator<Item = S> + ExactSizeIterator<Item = S>;
type EnumeratedSlice<'a, T> = (usize, &'a Vec<T>);
type EnumeratedSliceMut<'a, T> = (usize, &'a mut Vec<T>);
type Block<T> = Vec<T>;
// type Block<'a, T> = Vec<&'a T>;

trait ThreeBlocks<T> = Iterator<Item = T>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Sudoku<T>(Vec<Vec<T>>);

impl<T: Clone + Default + std::fmt::Debug> Sudoku<T> {
    fn from(entries: Vec<Vec<T>>) -> Self {
        // assert_eq!(entries.len(), 81);
        Sudoku(entries)
    }
    fn rows(
        &self,
    ) -> impl ExactSizeIterator<Item = impl Row<&'_ T>> + DoubleEndedIterator<Item = impl Row<&'_ T>>
    {
        self.0
            .chunks(3)
            .map(|chunk| exactly(chunk.iter().flatten(), 9))
    }
    fn rows_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = impl Row<&'_ mut T>>
           + DoubleEndedIterator<Item = impl Row<&'_ mut T>> {
        self.0
            .chunks_mut(3)
            .map(|chunk| exactly(chunk.iter_mut().flatten(), 9))
    }
    fn three_block_chunks(&self) -> impl Iterator<Item = impl ThreeBlocks<EnumeratedSlice<'_, T>>> {
        self.0
            .chunks(9)
            // &[Vec<T;3>;9]
            .map(|chunk| chunk.iter().enumerate())
    }
    fn three_block_chunks_mut(
        &mut self,
    ) -> impl Iterator<Item = impl ThreeBlocks<EnumeratedSliceMut<'_, T>>> {
        self.0
            .chunks_mut(9)
            // &[Vec<T;3>;9]
            .map(|chunk| chunk.iter_mut().enumerate())
    }
    fn blocks(&self) -> impl Iterator<Item = Block<&'_ T>> {
        // Iter<Item = Iter<Item = (usize, &Vec<T; 3>); 3>; ?>
        self.three_block_chunks()
            // Iter<Item = (usize, &Vec<T; 3>); 9>
            // maps every chunk to an iterator over blocks
            .map(|enum_chunk| {
                Iter::new(enum_chunk.scan(vec![], |visited, (counter, slice)| {
                    visited.push((counter, slice));
                    if counter != 8 {
                        None
                    } else {
                        let iter = once(remove_any(&[0, 3, 6], visited))
                            .chain(once(remove_any(&[1, 4, 7], visited)))
                            .chain(once(remove_any(&[2, 5, 8], visited)));
                        Some(iter)
                    }
                }))
                .step_by(8)
                .skip(1)
            })
            .flatten()
            .flatten()
    }
    fn blocks_mut(&mut self) -> impl Iterator<Item = Block<&'_ mut T>> {
        // Iter<Item = Iter<Item = (usize, &Vec<T; 3>); 3>; ?>
        self.three_block_chunks_mut()
            // Iter<Item = (usize, &Vec<T; 3>); 9>
            // maps every chunk to an iterator over blocks
            .map(|enum_chunk| {
                Iter::new(enum_chunk.scan(vec![], |visited, (counter, slice)| {
                    visited.push((counter, slice));
                    if counter != 8 {
                        None
                    } else {
                        let iter = once(remove_any(&[0, 3, 6], visited))
                            .chain(once(remove_any(&[1, 4, 7], visited)))
                            .chain(once(remove_any(&[2, 5, 8], visited)));
                        Some(iter)
                    }
                }))
                .step_by(8)
                .skip(1)
            })
            .flatten()
            .flatten()
    }
}

fn remove_any<'a, T: 'a, H: IntoIterator<Item = T>>(
    t: &[usize],
    visited: &mut Vec<(usize, H)>,
) -> Block<T> {
    visited
        .drain_filter(|(n, _)| t.iter().any(|elt| elt == n))
        .map(|(_, slice)| slice.into_iter())
        .flatten()
        .collect::<Vec<_>>()
}

fn consume<T, I: Iterator<Item = T>>(amount: usize, mut iter: I) -> Vec<Option<T>> {
    let mut ret = vec![];
    for _ in 0..amount {
        ret.push(iter.next());
    }
    ret
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;
    // #[test]
    // fn test_unstable() {
    //     let mut unstable = Flaky::new();
    //     // dbg!(unstable.next());
    //     // dbg!(unstable.next());
    //     let mut skip = unstable.skip(1);
    //     dbg!(skip.next());
    //     dbg!(skip.next());
    // }
    #[test]
    fn test_iterator_blocks() {
        let default: Vec<u8> = (1..=81).collect();
        let vecvec = default
            .chunks(3)
            .map(|chunk| chunk.iter().cloned().collect::<Vec<_>>()) //;
            .collect::<Vec<_>>();
        let s = Sudoku::from(vecvec);
        for block in s.blocks() {
            dbg!(block);
        }
        // let mut enum_chunks = s.enum_chunks().collect::<Vec<_>>();
        // let first: Vec<_> = enum_chunks.remove(0).collect();
        // let mut potential_blocks = Iter::new(collect_into_blocks(first.iter().cloned()));
        // for _ in 0..1 {
        //     let blocks: Vec<_> = match potential_blocks.next() {
        //         None => {
        //             println!("None");
        //             continue;
        //         }
        //         Some(blocks) => blocks.collect(),
        //     };
        //     // for block in blocks {
        //     dbg!(blocks);
        //     // }
        // }
    }

    #[test]
    fn test_block() {}
    // #[test]
    // fn test_valid_blocks() {
    //     let default: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    //     let blocks: Vec<_> = vec![Block::from(default); 9];
    //     let s = Sudoku::from(blocks);
    //     let t = s.transpose();
    //     assert!(s.validate_blocks());
    //     assert!(t.validate_blocks());
    // }
    // #[test]
    // fn test_valid_blocks() {
    //     let default: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    //     let blocks: Vec<_> = vec![Block::from(default); 9];
    //     let s = Sudoku::from(blocks);
    //     let t = s.transpose();
    //     assert!(s.validate_blocks());
    //     assert!(t.validate_blocks());
    // }
    // #[test]
    // fn test_valid_solution() {
    //     let n1 = vec![5, 3, 4, 6, 7, 2, 1, 9, 8];
    //     let n2 = vec![6, 7, 8, 1, 9, 5, 3, 4, 2];
    //     let n3 = vec![9, 1, 2, 3, 4, 8, 5, 6, 7];
    //     let n4 = vec![8, 5, 9, 4, 2, 6, 7, 1, 3];
    //     let n5 = vec![7, 6, 1, 8, 5, 3, 9, 2, 4];
    //     let n6 = vec![4, 2, 3, 7, 9, 1, 8, 5, 6];
    //     let n7 = vec![9, 6, 1, 2, 8, 7, 3, 4, 5];
    //     let n8 = vec![5, 3, 7, 4, 1, 9, 2, 8, 6];
    //     let n9 = vec![2, 8, 4, 6, 3, 5, 1, 7, 9];
    //     let raw = vec![n1, n2, n3, n4, n5, n6, n7, n8, n9];
    //     let mut blocks = vec![];
    //     blocks.extend(raw.into_iter().map(|elt| Block::from(elt)));
    //     let valid = Sudoku::from(blocks);
    //     let t_valid = valid.transpose();

    //     assert!(valid.validate());
    //     assert!(t_valid.validate());
    // }
}

#[derive(Debug, Clone)]
struct Few<T> {
    amount: usize,
    counter: usize,
    element: T,
}

impl<T: Clone> Iterator for Few<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.counter < self.amount {
            self.counter += 1;
            Some(self.element.clone())
        } else {
            None
        }
    }
}
impl<T: Clone> DoubleEndedIterator for Few<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

fn few<T>(element: T, amount: usize) -> Few<T> {
    Few {
        amount,
        counter: 0,
        element,
    }
}

impl<T: Clone> ExactSizeIterator for Few<T> {
    fn len(&self) -> usize {
        self.amount - (self.counter + 1)
    }
}

#[derive(Debug, Clone)]
struct Exact<I> {
    len: usize,
    iter: I,
}

impl<I: Iterator> Iterator for Exact<I> {
    type Item = <I as Iterator>::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.len > 0 {
            self.len -= 1;
            self.iter.next()
        } else {
            None
        }
    }
}
impl<I: DoubleEndedIterator> DoubleEndedIterator for Exact<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len > 0 {
            self.len -= 1;
            self.iter.next_back()
        } else {
            None
        }
    }
}

fn exactly<I: Iterator>(iter: I, len: usize) -> Exact<I> {
    Exact { len, iter }
}

impl<I: Iterator> ExactSizeIterator for Exact<I> {
    fn len(&self) -> usize {
        self.len
    }
}

struct Flaky {
    next: bool,
}

impl Flaky {
    fn new() -> Self {
        Flaky { next: false }
    }
}
impl Iterator for Flaky {
    type Item = ();
    fn next(&mut self) -> Option<()> {
        self.next = !self.next;
        if !self.next {
            Some(())
        } else {
            None
        }
    }
    // fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
    //     loop {
    //         let ret = self.next();
    //         if n == 0 {
    //             return ret;
    //         }
    //         n -= 1;
    //     }
    // }
}

struct Iter<I> {
    iter: I,
}

impl<I: Iterator> Iter<I> {
    fn new(iter: I) -> Self {
        Iter { iter }
    }
}
impl<I: Iterator> Iterator for Iter<I> {
    type Item = <I as Iterator>::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        loop {
            let ret = self.next();
            if n == 0 {
                return ret;
            }
            n -= 1;
        }
    }
}
