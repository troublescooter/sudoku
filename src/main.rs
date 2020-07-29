#![allow(dead_code)]
#![allow(unused_imports)]
use std::fmt::Debug;
use std::num::ParseIntError;
use thiserror::Error;

#[derive(Copy, Clone, PartialEq, Eq)]
struct Cell(u16);
impl Default for Cell {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Cell {
    fn fmt(&self, dest: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value() {
            None => write!(dest, " "),
            Some(val) => write!(dest, "{}", val),
        }
    }
}
impl std::fmt::Debug for Cell {
    fn fmt(&self, dest: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value() {
            None => write!(dest, "{:b}", self.0),
            Some(val) => write!(dest, "{}", val),
        }
    }
}

impl Cell {
    fn new() -> Self {
        Self(0b1_1111_1111)
    }

    fn is_final(&self) -> bool {
        self.0 & (self.0 - 1) == 0
    }

    fn set(&mut self, value: Value) {
        self.0 = value.mask();
    }

    fn contains(&self, value: Value) -> bool {
        self.0 & value.mask() != 0
    }

    fn remove(&mut self, value: Value) {
        if !self.is_final() {
            self.0 &= !value.mask();
        }
    }

    fn value(&self) -> Option<Value> {
        if !self.is_final() {
            return None;
        }
        let n = match self.0 {
            1 => 1,
            2 => 2,
            4 => 3,
            8 => 4,
            16 => 5,
            32 => 6,
            64 => 7,
            128 => 8,
            256 => 9,
            _ => return None,
        };
        Some(Value::new(n).expect("valid value; qed"))
    }
}
#[derive(Clone)]
struct Sudoku([Cell; 81]);
impl Sudoku {
    fn new() -> Self {
        Self([Default::default(); 81])
    }
    fn blocks(&self) -> Blocks<'_> {
        let mut blocks = Blocks::new();
        for (c, elt) in self.0.iter().enumerate() {
            let block = which_block(c);
            blocks.add_to_block(block, elt);
        }
        blocks
    }
    fn blocks_mut(&mut self) -> BlocksMut<'_> {
        let mut blocks_mut = BlocksMut::new();
        for (c, elt) in self.0.iter_mut().enumerate() {
            let block = which_block(c);
            blocks_mut.add_to_block(block, elt);
        }
        blocks_mut
    }
    fn rows_mut(&mut self) -> RowsMut<'_> {
        let mut rows_mut = RowsMut::new();
        for (c, elt) in self.0.iter_mut().enumerate() {
            let row = which_row(c);
            rows_mut.add_to_row(row, elt);
        }
        rows_mut
    }
    fn rows(&self) -> Rows<'_> {
        let mut rows = Rows::new();
        for (c, elt) in self.0.iter().enumerate() {
            let row = which_row(c);
            rows.add_to_row(row, elt);
        }
        rows
    }
    fn columns_mut(&mut self) -> ColumnsMut<'_> {
        let mut columns_mut = ColumnsMut::new();
        for (c, elt) in self.0.iter_mut().enumerate() {
            let col = which_column(c);
            columns_mut.add_to_column(col, elt);
        }
        columns_mut
    }
    fn get(&self, row: usize, col: usize) -> Option<&Cell> {
        self.0.get(linearise((row, col)))
    }
    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Cell> {
        self.0.get_mut(linearise((row, col)))
    }
    fn valid(&self) -> bool {
        for i in 0..9 {
            let row = self.validate_group(row(i));
            let col = self.validate_group(column(i));
            let block = self.validate_group(block(i));
            if !row || !col || !block {
                return false;
            }
        }
        true
    }
    fn not_invalid_group(&self, iter: impl Iterator<Item = (usize, usize)>) -> bool {
        let mut bitfield = 0u16;
        for (row, col) in iter {
            let cell = self.get(row, col).unwrap();
            match cell.value() {
                None => {
                    continue;
                }
                Some(_) => {
                    if bitfield & cell.0 != 0 {
                        return false;
                    } else {
                        bitfield |= cell.0;
                    }
                }
            }
        }
        true
    }
    fn validate_group(&self, iter: impl Iterator<Item = (usize, usize)>) -> bool {
        let mut check = 0;
        for (row, col) in iter {
            let cell = self.get(row, col).expect("There should be a value here");
            if !cell.is_final() {
                return false;
            }
            check |= cell.0
        }
        check == 0b1_1111_1111
    }
}
impl Default for Sudoku {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Sudoku {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for x in 0..9 {
            for y in 0..9 {
                write!(f, "{:?} ", self.get(x, y).unwrap())?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Sudoku {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for x in 0..9 {
            for y in 0..9 {
                write!(f, "{}", self.get(x, y).unwrap())?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::str::FromStr for Sudoku {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        let mut sudoku = Sudoku::new();
        // let space = " ".to_string();
        for (r, row) in string.split('\n').enumerate() {
            for (c, ch) in row.chars().enumerate() {
                if ch == ' ' {
                    continue;
                } else {
                    let value = ch.to_string().parse()?;
                    sudoku.get_mut(r, c).unwrap().set(value);
                }
            }
        }
        Ok(sudoku)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Value(u8);

impl Value {
    fn new(value: u8) -> Result<Self, Error> {
        if value >= 1 && value <= 9 {
            Ok(Self(value - 1))
        } else {
            Err(Error::ValueOutOfRange)
        }
    }

    fn mask(&self) -> u16 {
        0b1 << self.0
    }
}

impl std::str::FromStr for Value {
    type Err = Error;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        Self::new(string.parse()?)
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0 + 1)
    }
}
#[derive(Debug, Error)]
enum Error {
    #[error("value out of range")]
    ValueOutOfRange,
    #[error(transparent)]
    ParseInt(#[from] ParseIntError),
}

fn linearise((row, col): (usize, usize)) -> usize {
    9 * row + col
}

fn row_column(index: usize) -> (usize, usize) {
    let col = index % 9;
    let row = (index - col) / 9;
    (row, col)
}

fn row(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..9).map(move |col| (n, col))
}
fn column(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..9).map(move |row| (row, n))
}
fn block(n: usize) -> impl Iterator<Item = (usize, usize)> {
    let first = 27 * (n/3) + 3*(n%3);
    (first..first + 3)
        .chain(first + 9..(first + 3) + 9)
        .chain(first + 18..(first + 3) + 18)
        .map(row_column)
}
fn which_block(idx: usize) -> usize {
    let (row, col) = row_column(idx);
    3 * (row / 3) + col / 3
}
fn which_row(idx: usize) -> usize {
    let (row, _) = row_column(idx);
    row
}
fn which_column(idx: usize) -> usize {
    let (_, col) = row_column(idx);
    col
}

#[derive(Debug)]
struct Blocks<'a>(Vec<Vec<&'a Cell>>);
#[derive(Debug)]
struct BlocksMut<'a>(Vec<Vec<&'a mut Cell>>);
impl<'a,> Blocks<'a> {
    fn new() -> Self {
        let mut empty = vec![];
        for _ in 0..9 {
            empty.push(vec![]);
        }
        Self(empty)
    }
    fn add_to_block(&mut self, block: usize, elt: &'a Cell) {
        self.0[block].push(elt);
    }
}
impl<'a> BlocksMut<'a> {
    fn new() -> Self {
        let mut empty = vec![];
        for _ in 0..9 {
            empty.push(vec![]);
        }
        Self(empty)
    }
    fn add_to_block(&mut self, block: usize, elt: &'a mut Cell) {
        self.0[block].push(elt);
    }
}
struct Rows<'a>(Vec<Vec<&'a Cell>>);
impl<'a> Rows<'a> {
    fn new() -> Self {
        let mut empty = vec![];
        for _ in 0..9 {
            empty.push(vec![]);
        }
        Self(empty)
    }
    fn add_to_row(&mut self, row: usize, elt: &'a Cell) {
        self.0[row].insert(0, elt);
    }
}
struct RowsMut<'a>(Vec<Vec<&'a mut Cell>>);
impl<'a> RowsMut<'a> {
    fn new() -> Self {
        let mut empty = vec![];
        for _ in 0..9 {
            empty.push(vec![]);
        }
        Self(empty)
    }
    fn add_to_row(&mut self, row: usize, elt: &'a mut Cell) {
        self.0[row].insert(0, elt);
    }
}
struct ColumnsMut<'a>(Vec<Vec<&'a mut Cell>>);
impl<'a> ColumnsMut<'a> {
    fn new() -> Self {
        let mut empty = vec![];
        for _ in 0..9 {
            empty.push(vec![]);
        }
        Self(empty)
    }
    fn add_to_column(&mut self, column: usize, elt: &'a mut Cell) {
        self.0[column].insert(0, elt);
    }
}
impl<'a> Iterator for Blocks<'a> {
    type Item = Vec<&'a Cell>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.len() != 0 {
            Some(self.0.remove(0))
        } else {
            None
        }
    }
}

impl<'a> Iterator for BlocksMut<'a> {
    type Item = Vec<&'a mut Cell>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.len() != 0 {
           Some(self.0.remove(0))
        } else {
            None
        }
    }
}
impl<'a> Iterator for RowsMut<'a> {
    type Item = Vec<&'a mut Cell>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.len() != 0 {
            Some(self.0.remove(0))
        } else {
            None
        }
    }
}
impl<'a> Iterator for Rows<'a> {
    type Item = Vec<&'a Cell>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.len() != 0 {
            Some(self.0.remove(0))
        } else {
            None
        }
    }
}
impl<'a> Iterator for ColumnsMut<'a> {
    type Item = Vec<&'a mut Cell>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

fn solve_first_order<'a,I: IntoIterator<Item = &'a mut Cell>>(refs: I) {
    let mut carry = 0_u16;
    for elt in refs {
        elt.0 &= !carry;
        if elt.is_final() {
            carry |= elt.0;
        }
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    fn map_to_integers<'a>(refs: Vec<&'a mut Cell>) -> Vec<u8> {
        refs.into_iter().map(|cell| &*cell).cloned().map(|cell| cell.value().unwrap().0 + 1).collect()
    }
    use super::*;

    #[test]
    fn test_cell() {
        let mut cell = Cell::new();
        for i in 2..=9 {
            let value = Value::new(i).unwrap();
            assert!(!cell.is_final());
            assert!(cell.value().is_none());
            assert!(cell.contains(value));
            cell.remove(value);
            assert!(!cell.contains(value));
        }
        assert!(cell.is_final());
        assert_eq!(cell.value(), Some(Value::new(1).unwrap()));

        let mut cell = Cell::new();
        cell.set(Value::new(3).unwrap());
        assert!(cell.is_final());
        assert_eq!(cell.value(), Some(Value::new(3).unwrap()));
    }

    #[test]
    fn test_blocks_mut() {
        let mut sudoku: Sudoku = "534678912\n\
                                  672195348\n\
                                  198342567\n\
                                  859761423\n\
                                  426853791\n\
                                  713924856\n\
                                  961537284\n\
                                  287419635\n\
                                  345286179"
            .parse()
            .unwrap();
        let mut blocks = sudoku.blocks_mut();
        println!("{:?}",blocks);
        assert_eq!(map_to_integers(blocks.next().unwrap()),vec![5,3,4,6,7,2,1,9,8]);
    }

    #[test]
    fn test_validate_sudoku() {
        let sudoku: Sudoku = "534678912\n\
                              672195348\n\
                              198342567\n\
                              859761423\n\
                              426853791\n\
                              713924856\n\
                              961537284\n\
                              287419635\n\
                              345286179"
            .parse()
            .unwrap();
        // println!("{}", sudoku);
        // assert!(sudoku.validate_group(column(0)));
        // assert!(sudoku.validate_group(column(1)));
        // assert!(sudoku.validate_group(column(2)));
        // println!("{:?}", sudoku.blocks().0[4]);
        // assert!(sudoku.validate_group(block(4)));
        assert!(sudoku.valid());

    }
    #[test]
    fn test_not_invalid_sudoku() {
        let sudoku: Sudoku = " 34678 12\n\
                              6721 5348\n\
                              19 342567\n\
                              859761 23\n\
                              4268 3791\n\
                              7 3924856\n\
                              9615372 4\n\
                              287419635\n\
                              345286179"
            .parse()
            .unwrap();
        println!("{}", sudoku);
        assert!(sudoku.not_invalid_group(column(0)));
    }
    #[test]
    fn test_solve_sudoku() {
        let mut sudoku: Sudoku = " 34678 12\n\
                              6721 5348\n\
                              19 342567\n\
                              859761 23\n\
                              4268 3791\n\
                              7 3924856\n\
                              9615372 4\n\
                              287419635\n\
                              345286179"
            .parse()
            .unwrap();
        for row in sudoku.rows_mut() {
            solve_first_order(row.into_iter().rev());
        }
        println!("{:?}", sudoku);
        for row in sudoku.rows_mut() {
            solve_first_order(row);
        }
        for row in sudoku.rows_mut() {
            solve_first_order(row.into_iter().rev());
        }
        for block in sudoku.blocks_mut() {
            solve_first_order(block);
        }
        for col in sudoku.columns_mut() {
            solve_first_order(col);
        }
        println!("{:?}", sudoku);
        assert!(sudoku.valid());
    }
    #[test]
    fn test_row_column() {
        assert_eq!(row_column(0),(0,0));
        assert_eq!(row_column(1),(0,1));
        assert_eq!(row_column(2),(0,2));
        assert_eq!(row_column(3),(0,3));
        assert_eq!(row_column(4),(0,4));
        assert_eq!(row_column(5),(0,5));
        assert_eq!(row_column(9),(1,0));
    }
    #[test]
    fn test_which_block() {
        assert_eq!(which_block(0),0);
        assert_eq!(which_block(10),0);
        assert_eq!(which_block(1),0);
    }
}
