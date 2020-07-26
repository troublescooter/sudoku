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
trait CompleteIterator<S> {}
impl<S, T: DoubleEndedIterator<Item = S> + ExactSizeIterator<Item = S>> CompleteIterator<S> for T {}
// type DoubleExact<T> = DoubleSidedIterator<Item= >

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
    // fn blocks(&self) -> impl DoubleEndedIterator<Item = impl CompleteIterator<(usize, &'_ T)>> {
    //     // fn blocks(&self) -> Vec<&T> {
    //     let index = vec![0, 1, 2, 9, 10, 11, 18, 19, 20];
    //     let mut index_rev: Vec<_> = index.iter().rev().collect();
    //     self.0
    //         .chunks(27)
    //         .enumerate()
    //         // (usize,&[T])
    //         .scan(
    //             // (visited chunks on a stack, current index to filter the chunks by)
    //             (vec![],(()index_rev.pop().unwrap()))
    //                 ,|(visited,index), (counter,slice)| {
    //                     visited.push(slice);
    //                     let iter =visited.drain_filter(|elt|  counter == **index || counter == **index + 3 || counter == **index + 6);
    //                     *index = index_rev.pop().unwrap();
    //                     Some(iter)
    //                 }
    //         )

    // .map(|(counter, slice)| {
    //     few(counter, 3).zip(
    //         slice.iter(),
    //     )
    // })
    // every iter
    // (iter((usize,&T)))
    // .scan(vec![], |visited, &(n, x)| {
    //     let i = index_rev.pop().expect("There should be enough elements!");
    //     visited.drain_filter(|&(en,elt)|  *en == i || *en == i + 3 || *en == i + 6)
    // });
    // }
    fn blocks(&self) -> impl Iterator<Item = impl Iterator<Item = &'_ T>> {
        // let remove = |t,visited: &mut Vec<(_,_)>| // -> ()
        // {
        //     let removed: Vec<_> = visited.drain_filter(|(&n,_)| n == t || n == t + 3 || n== t+6 ).collect();
        //     removed.into_iter().map(|(_,slice)| slice)
        // };
        self.0
            .chunks(27)
            // three_whole_blocks
            .map(|three_blocks| {
                three_blocks
                    .chunks(3)
                    .enumerate()
                    .scan(vec![], |visited, (counter, slice)| {
                        visited.push((counter, slice));
                        if counter < 8 {
                            None
                        } else {
                            let iter = once(removed(0, visited))
                                .chain(once(removed(1, visited)))
                                .chain(once(removed(2, visited)));
                            // let t = counter - 6;
                            Some(iter)
                        }
                    })
                    .flatten()
            })
            .flatten()
    }
}

fn collect_into_blocks<T: std::fmt::Debug>(
    three_whole_blocks: &[T],
    // ) -> impl Iterator<Item = impl Iterator<Item = &'_ T>> {
) -> impl Iterator<Item = impl Iterator<Item = impl Iterator<Item = &'_ T>>> {
    // let remove = |t,visited: &mut Vec<(_,_)>| // -> ()
    // {
    //     let removed: Vec<_> = visited.drain_filter(|(&n,_)| n == t || n == t + 3 || n== t+6 ).collect();
    //     removed.into_iter().map(|(_,slice)| slice)
    // };
    three_whole_blocks
        .chunks(3)
        .enumerate()
        .scan(vec![], |visited, (counter, slice)| {
            visited.push((counter, slice));
            if counter < 8 {
                None
            } else {
                let iter = once(removed(0, visited))
                    .chain(once(removed(1, visited)))
                    .chain(once(removed(2, visited)));
                // let t = counter - 6;
                Some(iter)
            }
        })
    // .map(|elt| elt.map(|x| dbg!(x)))
    // .map(|iter|iter.flatten())
    // .flatten()
}
fn removed<'a, T>(
    t: usize,
    visited: &mut Vec<(usize, &'a [T])>,
) -> impl DoubleEndedIterator<Item = &'a T> {
    // let remove = |t,visited: &mut Vec<(_,_)>| // -> ()
    let removed: Vec<_> = visited
        .drain_filter(|(n, _)| *n == t || *n == t + 3 || *n == t + 6)
        .collect();
    removed.into_iter().map(|(_, slice)| slice.iter()).flatten()
}

fn consume<T, I: Iterator<Item = T>>(amount: usize, mut iter: I) -> Vec<Option<T>> {
    let mut ret = vec![];
    for _ in 0..amount {
        ret.push(iter.next());
    }
    ret
}

fn main() {}

fn recursive_consume<T, I: Iterator<Item = T>>(
    to_consume: Vec<Option<I>>,
) -> Vec<Option<Vec<Option<T>>>> {
    to_consume
        .into_iter()
        .map(|maybe_iter| maybe_iter.map(|iter| consume(3, iter)))
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_iterator_blocks() {
        let default: Vec<u8> = (1..=81).collect();
        // let s = Sudoku::from(default);
        // let mut visited = vec![];
        // let mut blocks = s.blocks();
        let three_blocks: Vec<_> = (1..=27).collect();
        let v = consume(3,collect_into_blocks(&three_blocks).map(|iter|consume(19,iter.map(|iter2|consume(13,iter2)))));
        dbg!(v);
    }

    // for three_whole_blocks in s.0.chunks(27) {
    //     for (counter, slice_of_three) in three_whole_blocks.chunks(3).enumerate() {
    //         visited.push((counter, slice_of_three));
    //         if counter < 6 {
    //             // return None;
    //         } else {
    //             let t = counter - 6;
    //             let v: Vec<_> = removed(t, &mut visited).collect();
    //             dbg!(v);
    //         }
    //     }
    // visited.
    // if counter < 6 {
    //     // return None;
    // } else {
    //     let t = counter - 6;
    //     let v : Vec<_>= removed(t,&mut visited).collect();
    //     dbg!(v);
    // }
    // let collect_slices_of_three_into_blocks(s.0.chunks(27).map(|chunk| chunk.chunks(3)));
    // for _ in 0..27 {

    //     dbg!(elt);
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
