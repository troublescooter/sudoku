#![allow(dead_code)]
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::num::ParseIntError;
use thiserror::Error;

const SUDOKU: &str = "534678912\n\
                      672195348\n\
                      198342567\n\
                      859761423\n\
                      426853791\n\
                      713924856\n\
                      961537284\n\
                      287419635\n\
                      345286179";

#[derive(Debug, Clone)]
struct Permutation(Vec<usize>);
#[derive(Debug, Clone)]
struct SymmetricGroup {
    size: usize,
}
#[derive(Debug, Clone)]
struct Sudokus;

impl Distribution<Permutation> for SymmetricGroup {
    fn sample<R: ?Sized + Rng>(&self, rng: &mut R) -> Permutation {
        let mut sigma: Vec<_> = (1..=self.size).collect();
        sigma.shuffle(rng);
        assert!(sigma.len() == 9);
        Permutation(sigma)
    }
}


#[derive(Debug, Error)]
struct UnsolvableWithRules;

impl std::fmt::Display for UnsolvableWithRules {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Debug>::fmt(self, f)
    }
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

#[derive(Copy, Clone, PartialEq, Eq)]
struct Cell(u16);
impl Default for Cell {
    fn default() -> Self {
        Self::new()
    }
}

enum Status {
    Finished,
    Continuing,
}
struct Choose {
    n: usize,
    status: Status,
    init: u16,
    mask: u16,
    max_mask: u16,
}

impl Choose {
    fn new(n: usize, init: u16) -> Self {
        let max_mask = if init == 0 {
            1u16 << (n - 1)
        } else {
            let mut shift = init;
            let mut c = 0;
            while shift >> 1 != 0 {
                shift >>= 1;
                c += 1;
            }
            1u16 << c
        };
        Choose {
            n,
            status: Status::Continuing,
            max_mask,
            mask: 1u16,
            init,
        }
    }
}

impl Iterator for Choose {
    type Item = u16;
    fn next(&mut self) -> Option<u16> {
        if let Status::Finished = self.status {
            return None;
        }
        let maybe_flipped = self.mask | self.init;
        if self.mask == self.max_mask {
            self.status = Status::Finished;
        };
        self.mask <<= 1;
        if maybe_flipped == self.init {
            // didn't flip bit, proceed to next
            self.next()
        } else {
            // did flip bit, return the result
            Some(maybe_flipped)
        }
    }
}

// all possible bit combinations of choosing k bits being 1 out of the lower 9 from a u16
fn choose(n: usize, k: usize) -> Box<dyn Iterator<Item = u16>> {
    if k == 0 {
        return Box::new(std::iter::once(0_u16));
    }
    Box::new(
        std::iter::repeat(n)
            .zip(choose(n, k - 1))
            .map(|(n, bitmask)| Choose::new(n, bitmask))
            .flatten(),
    )
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

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Cell) -> Option<Ordering> {
        let lhs = self.0;
        let rhs = other.0;
        if self == other {
            return Some(Ordering::Equal);
        }
        if lhs & rhs == lhs {
            return Some(Ordering::Less);
        }
        if lhs & rhs == rhs {
            return Some(Ordering::Greater);
        }
        None
    }
}

impl Cell {
    fn permute(&mut self, sigma: &Permutation) {
        let mut replacement = 0u16;
        let bitmask = 1u16;
        for n in 0..9 {
            if bitmask << n & self.0 != 0 {
                replacement |= bitmask << (sigma.0[n] - 1);
            }
        }
        self.0 = replacement;
    }
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
// if there exist exactly k cells that could contain any of k elements, then no other cell can contain these
fn matches_rule1(n: usize, k: usize, row: Vec<&Cell>) -> Vec<Cell> {
    let choose_k = choose(n, k).map(Cell);
    choose_k
        .map(|choice| {
            (
                choice,
                row.iter().rev().try_fold(0usize, |acc, &elt| {
                    if let Some(Ordering::Equal) = elt.partial_cmp(&&choice) {
                        Some(acc + 1)
                    } else {
                        Some(acc)
                    }
                }),
            )
        })
        .filter_map(|(cell, x)| match x {
            Some(number) if number == k => Some(cell),
            _ => None,
        })
        .collect::<Vec<_>>()
}
// output are the cells representing subsets S s.t.
// \E C' subset Row : |S| = |C'| = k
// /\ \A e in C' : Cell(S) subset e
// /\ \A ec in Row \ C' : Cell(S) \cap ec == {}
// so cells that representing k elements that can only be placed inside k fields in a row
// all elements not in these k elements will be removed from every element in C'
fn matches_rule2(n: usize, k: usize, row: Vec<&Cell>) -> Vec<Cell> {
    let choose_k = choose(n, k).map(Cell);
    choose_k
        .map(|choice| {
            (
                choice,
                row.iter().try_fold(0usize, |acc, &elt| {
                    if let Some(Ordering::Equal) | Some(Ordering::Greater) =
                        elt.partial_cmp(&&choice)
                    {
                        Some(acc + 1)
                    } else if elt.0 & choice.0 != 0 {
                        None
                    } else {
                        Some(acc)
                    }
                }),
            )
        })
        .filter_map(|(cell, x)| match x {
            Some(number) if number == k => Some(cell),
            _ => None,
        })
        .collect::<Vec<_>>()
}
#[derive(Clone)]
struct Sudoku([Cell; 81]);

impl PartialEq for Sudoku {
    fn eq(&self, other: &Sudoku) -> bool {
        self.iter().eq(other.iter())
    }
}
impl Sudoku {
    // returns a Sudoku with this many erased fields
    fn erased(amount: usize) -> Self {
        let mut sudoku = Sudoku::random();
        for _ in 0..amount {
            sudoku.erase();
        }
        sudoku
    }
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let perm = SymmetricGroup { size: 9 }.sample(&mut rng);
        let mut sudoku: Sudoku = SUDOKU.parse().unwrap();
        sudoku.permute(&perm);
        sudoku
    }
    // erase one more field at random
    fn erase(&mut self) {
        let dist = rand::distributions::Uniform::new(0,81);
        let mut rng = rand::thread_rng();
        loop {
        let index = dist.sample(&mut rng);
            if self.0[index] == Cell::default() {
                continue;
            } else {
                self.0[index] = Cell::default();
                return;
            }
        }
    }
    fn permute(&mut self, sigma: &Permutation) {
        for cell in self.0.iter_mut() {
            cell.permute(sigma);
        }
    }
    fn iter(&self) -> impl Iterator<Item = &Cell> {
        self.0.iter()
    }
    fn solve_order(&mut self, k: usize) {
        let iter_fn = [Sudoku::rows, Sudoku::columns, Sudoku::blocks];
        let iter_mut_fn = [Sudoku::rows_mut, Sudoku::columns_mut, Sudoku::blocks_mut];
        for (it_fn, it_mut_fn) in iter_fn.iter().zip(&iter_mut_fn) {
            let mut patterns = vec![];
            for part in it_fn(self) {
                patterns.push(matches_rule1(9, k, part));
            }
            for (part, pattern) in it_mut_fn(self).zip(patterns) {
                for elt in part.into_iter().rev() {
                    for p in pattern.iter() {
                        // remove all elements not in p1 from the elements that don't match p1
                        if p != elt {
                            *elt = Cell(elt.0 & !p.0);
                        }
                    }
                }
            }
            let mut patterns = vec![];
            for part in it_fn(self) {
                patterns.push(matches_rule2(9, k, part));
            }
            for (part, pattern) in it_mut_fn(self).zip(patterns) {
                for elt in part.into_iter().rev() {
                    for p in pattern.iter() {
                        // remove all other elements but the ones in pattern p2
                        if p <= elt {
                            *elt = Cell(elt.0 & p.0);
                        }
                    }
                }
            }
        }
    }
    fn solve(&mut self) -> Result<(), UnsolvableWithRules> {
        while !self.valid() {
            let before = self.clone();
            for k in 1..=9 {
                self.solve_order(k);
                if *self != before {
                    break;
                }
            }
            if *self == before {
                return Err(UnsolvableWithRules);
            }
        }
        Ok(())
    }
    fn new() -> Self {
        Self([Default::default(); 81])
    }
    fn blocks(&self) -> Blocks<&'_ Cell> {
        let mut blocks = Blocks::new();
        for (c, elt) in self.0.iter().enumerate() {
            blocks.add_with_oracle(
                |_| {
                    let (row, col) = row_column(c);
                    3 * (row / 3) + col / 3
                },
                elt,
            );
        }
        blocks
    }
    fn blocks_mut(&mut self) -> Blocks<&'_ mut Cell> {
        let mut blocks_mut = Blocks::new();
        for (c, elt) in self.0.iter_mut().enumerate() {
            blocks_mut.add_with_oracle(
                |_| {
                    let (row, col) = row_column(c);
                    3 * (row / 3) + col / 3
                },
                elt,
            );
        }
        blocks_mut
    }
    fn rows(&self) -> Rows<&'_ Cell> {
        let mut rows = Rows::new();
        for (c, elt) in self.0.iter().enumerate() {
            rows.add_with_oracle(
                |_| {
                    let (row, _) = row_column(c);
                    row
                },
                elt,
            );
        }
        rows
    }
    fn rows_mut(&mut self) -> Rows<&'_ mut Cell> {
        let mut rows_mut = Rows::new();
        for (c, elt) in self.0.iter_mut().enumerate() {
            rows_mut.add_with_oracle(
                |_| {
                    let (row, _) = row_column(c);
                    row
                },
                elt,
            );
        }
        rows_mut
    }
    fn columns(&self) -> Columns<&'_ Cell> {
        let mut columns = Columns::new();
        for (c, elt) in self.0.iter().enumerate() {
            columns.add_with_oracle(
                |_| {
                    let (_, col) = row_column(c);
                    col
                },
                elt,
            );
        }
        columns
    }
    fn columns_mut(&mut self) -> Columns<&'_ mut Cell> {
        let mut columns_mut = Columns::new();
        for (c, elt) in self.0.iter_mut().enumerate() {
            columns_mut.add_with_oracle(
                |_| {
                    let (_, col) = row_column(c);
                    col
                },
                elt,
            );
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
    fn not_invalid(&self) -> bool {
        for i in 0..9 {
            let row = self.not_invalid_group(row(i));
            let col = self.not_invalid_group(column(i));
            let block = self.not_invalid_group(block(i));
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
        for (r, row) in string.split('\n').enumerate() {
            for (c, ch) in row.chars().enumerate() {
                if ch == 'X' {
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
    let first = 27 * (n / 3) + 3 * (n % 3);
    (first..first + 3)
        .chain(first + 9..(first + 3) + 9)
        .chain(first + 18..(first + 3) + 18)
        .map(row_column)
}

type Rows<T> = GroupsOf<T>;
type Blocks<T> = GroupsOf<T>;
type Columns<T> = GroupsOf<T>;

#[derive(Clone, Debug)]
struct GroupsOf<T>(Vec<Vec<T>>);
impl<T> GroupsOf<T> {
    fn new() -> Self {
        let mut empty = vec![];
        for _ in 0..9 {
            empty.push(vec![]);
        }
        Self(empty)
    }
    fn add_with_oracle(&mut self, oracle: impl Fn(&T) -> usize, elt: T) {
        let group = oracle(&elt);
        self.0[group].push(elt);
    }
}
impl<T> Iterator for GroupsOf<T> {
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            None
        } else {
            Some(self.0.remove(0))
        }
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    #[test]
    fn test_generate_sudoku() {
        for _ in 0..10 {
            assert!(Sudoku::random().valid());
        }
    }
    #[test]
    fn test_generate_erased_sudoku() {
        for _ in 0..10 {
            assert!(Sudoku::erased(6).not_invalid());
        }
    }
    #[test]
    fn test_eq() {
        let s1 = Sudoku::default();
        let s2 = Sudoku::default();
        assert_eq!(s1, s2);
        let s2: Sudoku = "123".parse().unwrap();
        assert!(!(s1 == s2));
    }
    #[test]
    fn test_rules() {
        let mut sudoku: Sudoku = "X34678X12\n\
                              6721X5348\n\
                              19X342567\n\
                              859761X23\n\
                              4268X3791\n\
                              7X3924856\n\
                              9615372X4\n\
                              287419635\n\
                              345286179"
            .parse()
            .unwrap();
        let row = vec![Cell(0b1111), Cell(0b1111), Cell(0b11), Cell(0b11)];
        assert_eq!(matches_rule1(4, 2, row.iter().collect()), vec![Cell(0b11)]);
        let mut row = vec![
            Cell(0b1_1111_1111),
            Cell(0b100),
            Cell(0b1000),
            Cell(0b10_0000),
            Cell(0b0100_0000),
            Cell(0b1000_0000),
            Cell(0b1_1111_1111),
            Cell(0b1),
            Cell(0b10),
        ];
        let matches = matches_rule1(9, 1, row.iter().collect());
        assert_eq!(
            matches,
            vec![
                Cell(0b1),
                Cell(0b10),
                Cell(0b100),
                Cell(0b1000),
                Cell(0b10_0000),
                Cell(0b0100_0000),
                Cell(0b1000_0000),
            ]
        );
        for elt in row.iter_mut().rev() {
            // let (pattern1,pattern2) = patterns_per_row.pop().unwrap();
            for p1 in matches.iter() {
                // remove all elements not in p1 from the elements that don't match p1
                if p1 != elt {
                    *elt = Cell(elt.0 & !p1.0);
                }
            }
        }
        // dbg!(row);
        sudoku.solve_order(1);
        sudoku.solve_order(1);
        assert!(sudoku.not_invalid_group(column(0)));
    }
    #[test]
    fn test_choose() {
        let chosen = choose(9, 2).collect::<Vec<_>>();
        for elt in chosen {
            println!("{:8b}", elt);
        }
    }
    fn map_to_integers(refs: Vec<&mut Cell>) -> Vec<u8> {
        refs.into_iter()
            .map(|cell| &*cell)
            .cloned()
            .map(|cell| cell.value().unwrap().0 + 1)
            .collect()
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
        let mut sudoku: Sudoku = SUDOKU.parse().unwrap();
        let mut blocks = sudoku.blocks_mut();
        assert_eq!(
            map_to_integers(blocks.next().unwrap()),
            vec![5, 3, 4, 6, 7, 2, 1, 9, 8]
        );
    }

    #[test]
    fn test_validate_sudoku() {
        let sudoku: Sudoku = SUDOKU.parse().unwrap();
        assert!(sudoku.valid());
    }
    #[test]
    fn test_not_invalid_sudoku() {
        let sudoku: Sudoku = "X34678X12\n\
                              6721X5348\n\
                              19X342567\n\
                              859761X23\n\
                              4268X3791\n\
                              7X3924856\n\
                              9615372X4\n\
                              287419635\n\
                              345286179"
            .parse()
            .unwrap();
        assert!(sudoku.not_invalid());
    }
    #[test]
    fn test_unsolvable_sudoku() {
        let mut sudoku: Sudoku = "XXXXXXX12\n\
                                  6721XX348\n\
                                  19X34X567\n\
                                  X59X6XX23\n\
                                  42X8XX791\n\
                                  7X392XX5X\n\
                                  96153X2X4\n\
                                  287X1X63X\n\
                                  XXXXXXX7X"
            .parse()
            .unwrap();
        assert!(sudoku.solve().is_err());
    }
    #[test]
    fn test_solve_sudoku() {
        let mut sudoku: Sudoku = "X34678X12\n\
                                  6721X5348\n\
                                  19X342567\n\
                                  X59X61X23\n\
                                  42X8X3791\n\
                                  7X3924X56\n\
                                  9615372X4\n\
                                  287X1963X\n\
                                  X45286X79"
            .parse()
            .unwrap();
        sudoku.solve().expect("Should be solveable.");
        assert!(sudoku.valid());
    }
    #[test]
    fn test_row_column() {
        assert_eq!(row_column(0), (0, 0));
        assert_eq!(row_column(1), (0, 1));
        assert_eq!(row_column(2), (0, 2));
        assert_eq!(row_column(3), (0, 3));
        assert_eq!(row_column(4), (0, 4));
        assert_eq!(row_column(5), (0, 5));
        assert_eq!(row_column(9), (1, 0));
    }
    #[test]
    fn test_which_block() {
        assert_eq!(which_block(0), 0);
        assert_eq!(which_block(10), 0);
        assert_eq!(which_block(1), 0);
    }
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
