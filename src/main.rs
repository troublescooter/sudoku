#![allow(dead_code)]
use std::iter::once;
use std::slice::Iter;
#[derive(Debug, Clone)]
enum Entry {
    None,
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
        Entry::None
    }
}

#[derive(Debug, Clone)]
struct Sudoku<T = Entry>(Vec<Block<T>>);
impl<T: Default + Clone> Sudoku<T> {
    fn iter(&self) -> impl Iterator<Item = &Block<T>> {
        self.0.iter()
    }
    fn chunks(&self, size: usize) -> std::slice::Chunks<Block<T>> {
        self.0.chunks(size)
    }
    fn new() -> Self {
        Sudoku(vec![])
    }
    fn from(blocks: Vec<Block<T>>) -> Self {
        Sudoku(blocks)
    }
    fn default() -> Self {
        Sudoku(vec![Block::<T>::default(); 9])
    }
    // fn blocks<'a>(&'a self) -> Blocks<'a, T> {
    //     Blocks {
    //         iter: self.0.iter(),
    //     }
    // }
    fn validate(&self) -> bool {
        // self.validate_rows() && self.validate_columns() && self.validate_subblocks()
        todo!()
    }
    // fn rows<'a>(&'a self) -> Rows<'a, T> {
    //     Rows {
    //         index: 0,
    //         block: self.blocks(),
    //         row_iter: self
    //             .blocks()
    //             .skip(0)
    //             .take(3)
    //             .map(|block| block.iter().skip(0).take(3))
    //             .flatten(),
    //     }
    // }
    // fn columns<'a>(&'a self) -> Columns<'a, T> {
    //     todo!()
    // }
}

#[derive(Debug, Clone)]
struct Block<T = Entry>(Vec<T>);
impl<T> Block<T> {
    fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
    fn new() -> Self {
        Block(vec![])
    }
}
impl<T: Default + Clone> Block<T> {
    fn default() -> Self {
        Block(vec![T::default(); 9])
    }
    fn from(entries: Vec<T>) -> Self {
        Block(entries)
    }
}

// #[derive(Debug, Clone)]
// struct Blocks<'a, T = Entry> {
//     iter: std::slice::Iter<'a, Block<T>>,
// }

// impl<'a, T> Iterator for Blocks<'a, T> {
//     type Item = &'a Block<T>;
//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next()
//     }
// }

// // struct for iterating over the rows, each yielding a row iterator
// #[derive(Debug, Clone)]
// struct Rows<'a, T = Entry> {
//     // vec of iterators yielding blocks accross the
//     block: Blocks<'a, T>,
//     // iterator over elements in one row
//     row_iter: impl Iterator<Item = &'a Block<T>>,
//     // indicates the current row
//     index: usize,
// }

// impl<'a, T: Clone> Iterator for Rows<'a, T> {
//     // type of row_iter
//     type Item = impl Iterator<Item = &'a Block<T>>;
//     fn next(&mut self) -> Option<Self::Item> {
//         self.index += 1;
//         if self.index == 9 {
//             return None;
//         }
//         if self.index % 3 == 0 {
//             self.block.next();
//             self.block.next();
//             self.block.next();
//             self.row_iter = self.block.clone().skip(0).take(3).map
//         } else {
//         }
//     }
// }

// struct for iterating over one row
// struct Row<'a, T = Entry> {
//     // iterator iterating over the blocks
//     iter_block: Blocks<'a, T>,
//     // current block
//     block: &'a Block<T>,
//     // indicates the current row
//     index: usize,
// }

// impl<'a,T: Clone> Iterator for Row<'a,T> {
//     type Item = &'a T;
//     fn next(&mut self) -> Option<Self::Item> {
//         let first = self.index % 3 * 3;
//         let iter = self.block.iter().skip(first).take(3);
//         match iter.next() {
//             ret @ Some(_) => {return ret},
//             None => {
//                 match self.iter_blocks.next() {
//                     Some(next_block) => {self.block = next_block},
//                     None => { None
//                 }

//             },
//         }
//     }
// }

// struct Columns<'a, T = Entry> {
//     block_iter: Blocks<'a, T>,
//     block_entries_enum: std::iter::Enumerate<std::slice::Iter<'a, T>>,
// }

fn main() {
    let blocks: Vec<Block<u8>> = vec![
        Block::from(vec![1]),
        Block::from(vec![2]),
        Block::from(vec![3]),
        Block::from(vec![4]),
        Block::from(vec![5]),
        Block::from(vec![6]),
        Block::from(vec![7]),
        Block::from(vec![8]),
        Block::from(vec![9]),
    ];
    let s = Sudoku::<u8>::from(blocks);

    // let per_row_blocks = |sudoku: &'a Sudoku<u8>| {};
    for block in per_row_blocks(&s) {
        println!("{:?}", block);
    }
    // println!("{}",3+1%3);
    //     for elt in s.blocks() {
    //         // dbg!(elt);
    //         // let x: () = elt;
    // }
}
// fn per_row_blocks<'a, T: Default + Clone>(
//     sudoku: &'a Sudoku<T>,
// ) -> std::iter::Map<
//     std::iter::Enumerate<std::slice::Chunks<'_, Block<T>>>,
//     for<'r> fn((usize, &'r [Block<T>])) -> [(usize, std::slice::Iter<'r, Block<T>>); 3],
// > {
//     sudoku.chunks(3).enumerate().map(subdivide)
// }
fn per_row_blocks<T: Default + Clone>(
    sudoku: &Sudoku<T>,
    // ) -> impl Iterator<Item = [(usize, std::slice::Iter<'_, Block<T>>); 3]> {
) -> impl Iterator<Item = (usize, Iter<'_, Block<T>>)> {
    sudoku
        .chunks(3)
        .enumerate()
        .map(|(count, row_blocks_slice)| {
            once((count * 3, row_blocks_slice.iter()))
                .chain(once((count * 3 + 1, row_blocks_slice.iter())))
                .chain(once((count * 3 + 2, row_blocks_slice.iter())))
        })
        .flatten()
}
// yields iterator of iterators over the blocks in a given row
// fn subdivide<'a, T>(
//     : (usize, &'a [Block<T>]),
//     // ) -> [(usize, std::slice::Iter<'a, Block<T>>); 3] {
// ) -> impl Iterator<Item = (usize, Iter<'_, Block<T>>)> {
// [
//     (count * 3, row_blocks_slice.iter()),
//     (count * 3 + 1, row_blocks_slice.iter()),
//     (count * 3 + 2, row_blocks_slice.iter()),
// ].iter()
// }
// fn flat<'a,T>(blocks_in_row:[(usize, std::slice::Iter<'a, Block<T>>); 3]) -> () {

// }
