#![allow(dead_code)]
#![feature(drain_filter)]
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Sudoku<T>(Vec<T>);

impl<T: Clone + Default> Sudoku<T> {
    fn new() -> Self {
        Sudoku(vec![T::default(); 81])
    }
    fn from(entries: Vec<T>) -> Self {
        assert_eq!(entries.len(), 81);
        Sudoku(entries)
    }
    fn iter(&self) -> impl DoubleEndedIterator<Item = &'_ T> {
        self.0.iter()
    }
    fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &'_ mut T> {
        self.0.iter_mut()
    }
    fn rows(&self) -> impl DoubleEndedIterator<Item = impl DoubleEndedIterator<Item = &'_ T>> {
        self.0.chunks(9).map(|chunk| chunk.iter())
    }
    fn blocks(&self) -> impl DoubleEndedIterator<Item = impl DoubleEndedIterator<Item = &'_ T>> {
        // fn blocks(&self) -> Vec<&T> {
        let index = vec![0, 1, 2, 9, 10, 11, 18, 19, 20];
        let mut index_rev: Vec<_> = index.iter().rev().collect();
        let enumerated_chunks: Vec<(usize, &[T])> = self
            .0
            .chunks(3)
            .enumerate()
            // Some((usize,&[T]))
            .map(|(n,slice)| repeat(n).zip(slice.iter()))
            .scan(vec![], |visited, &(n, x)| {
                let i = index_rev.pop().expect("There should be enough elements!");
                visited.drain_filter(|&(en,elt)|  *en == i || *en == i + 3 || *en == i + 6)
            });
        let mut blocks = vec![];
        for &i in index.iter() {
            blocks.push(
                enumerated_chunks
                    .iter()
                    .filter(|&(n, chunk)| *n == i || *n == i + 3 || *n == i + 6)
                    .map(|(_, chunk)| *chunk)
                    .collect::<Vec<_>>(),
            )
        }

        blocks
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_valid_blocks() {
        let default: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let blocks: Vec<_> = vec![Block::from(default); 9];
        let s = Sudoku::from(blocks);
        let t = s.transpose();
        assert!(s.validate_blocks());
        assert!(t.validate_blocks());
    }
    #[test]
    fn test_valid_solution() {
        let n1 = vec![5, 3, 4, 6, 7, 2, 1, 9, 8];
        let n2 = vec![6, 7, 8, 1, 9, 5, 3, 4, 2];
        let n3 = vec![9, 1, 2, 3, 4, 8, 5, 6, 7];
        let n4 = vec![8, 5, 9, 4, 2, 6, 7, 1, 3];
        let n5 = vec![7, 6, 1, 8, 5, 3, 9, 2, 4];
        let n6 = vec![4, 2, 3, 7, 9, 1, 8, 5, 6];
        let n7 = vec![9, 6, 1, 2, 8, 7, 3, 4, 5];
        let n8 = vec![5, 3, 7, 4, 1, 9, 2, 8, 6];
        let n9 = vec![2, 8, 4, 6, 3, 5, 1, 7, 9];
        let raw = vec![n1, n2, n3, n4, n5, n6, n7, n8, n9];
        let mut blocks = vec![];
        blocks.extend(raw.into_iter().map(|elt| Block::from(elt)));
        let valid = Sudoku::from(blocks);
        let t_valid = valid.transpose();

        assert!(valid.validate());
        assert!(t_valid.validate());
    }
}
