//! Delay pattern masking for 4-codebook architecture.
//!
//! MusicGen uses a delay pattern for parallel generation of 4 EnCodec codebooks.
//! This pattern ensures causality by applying progressive delays to each codebook:
//! - Codebook 0: no delay
//! - Codebook 1: 1 token delay
//! - Codebook 2: 2 token delay
//! - Codebook 3: 3 token delay

/// Delay pattern mask for N codebooks.
///
/// Manages token sequences with the delay pattern required for MusicGen's
/// 4-codebook parallel generation.
#[derive(Debug)]
pub struct DelayPatternMaskIds<const N: usize> {
    batches: [Vec<i64>; N],
}

impl<const N: usize> Default for DelayPatternMaskIds<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> DelayPatternMaskIds<N> {
    /// Creates a new empty delay pattern mask.
    pub fn new() -> Self {
        assert!(N > 0, "N needs to be greater than 0");
        Self {
            batches: [(); N].map(|()| vec![]),
        }
    }

    /// Pushes a new set of token IDs for all codebooks.
    ///
    /// # Panics
    ///
    /// Panics if the iterator does not yield exactly N token IDs.
    pub fn push(&mut self, token_ids: impl IntoIterator<Item = i64>) {
        let mut i = 0;
        for token_id in token_ids.into_iter() {
            assert!(i < N, "Expected exactly {N} token_ids");
            self.batches[i].push(token_id);
            i += 1;
        }
        assert_eq!(i, N, "Expected exactly {N} token_ids");
    }

    /// Returns the last token for each codebook with delay pattern applied.
    ///
    /// The delay pattern applies padding tokens to codebooks that haven't
    /// accumulated enough tokens yet:
    /// ```text
    ///   0 1 2 3 4 5 6 7 8 9 10
    /// 0 x x x x x x x x x x ...
    /// 1 P x x x x x x x x x ...
    /// 2 P P x x x x x x x x ...
    /// 3 P P P x x x x x x x ...
    /// ```
    pub fn last_delayed_masked(&self, pad_token_id: i64) -> [i64; N] {
        let seq_len = self.batches[0].len();
        let mut result = [0; N];
        for (i, item) in result.iter_mut().enumerate() {
            if (seq_len as i64 - i as i64) <= 0 {
                *item = pad_token_id
            } else {
                *item = *self.batches[i].last().expect("There are no input_ids");
            }
        }
        result
    }

    /// Returns the last de-delayed diagonal set of tokens.
    ///
    /// This extracts tokens from the diagonal pattern avoiding padding:
    /// ```text
    ///   0 1 2 3 4 5 6 7 8 9
    /// 0 x x x x x x x P P P
    /// 1 P x x x x x x x P P
    /// 2 P P x x x x x x x P
    /// 3 P P P x x x x x x x
    /// ```
    /// Returns None until enough tokens have been accumulated (N tokens).
    pub fn last_de_delayed(&self) -> Option<[i64; N]> {
        if self.batches[0].len() < N {
            return None;
        }
        let mut result = [0; N];
        for (i, item) in result.iter_mut().enumerate() {
            *item = self.batches[i][self.batches[i].len() - N + i]
        }
        Some(result)
    }

    /// Returns the number of tokens in the first codebook.
    pub fn len(&self) -> usize {
        self.batches[0].len()
    }

    /// Returns true if no tokens have been added yet.
    pub fn is_empty(&self) -> bool {
        self.batches[0].is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_delay_pattern() {
        let pattern = DelayPatternMaskIds::<4>::new();
        assert!(pattern.is_empty());
        assert_eq!(pattern.len(), 0);
    }

    #[test]
    fn last_delayed_masked() {
        let mut input_ids = DelayPatternMaskIds::<4>::new();
        assert_eq!(input_ids.last_delayed_masked(0), [0, 0, 0, 0]);
        input_ids.push([1, 2, 3, 4]);
        assert_eq!(input_ids.last_delayed_masked(0), [1, 0, 0, 0]);
        input_ids.push([5, 6, 7, 8]);
        assert_eq!(input_ids.last_delayed_masked(0), [5, 6, 0, 0]);
        input_ids.push([9, 10, 11, 12]);
        assert_eq!(input_ids.last_delayed_masked(0), [9, 10, 11, 0]);
        input_ids.push([13, 14, 15, 16]);
        assert_eq!(input_ids.last_delayed_masked(0), [13, 14, 15, 16]);
        input_ids.push([17, 18, 19, 20]);
        assert_eq!(input_ids.last_delayed_masked(0), [17, 18, 19, 20]);
    }

    #[test]
    fn last_de_delayed() {
        let mut input_ids = DelayPatternMaskIds::<4>::new();
        assert_eq!(input_ids.last_de_delayed(), None);
        input_ids.push([1, 2, 3, 4]);
        assert_eq!(input_ids.last_de_delayed(), None);
        input_ids.push([5, 6, 7, 8]);
        assert_eq!(input_ids.last_de_delayed(), None);
        input_ids.push([9, 10, 11, 12]);
        assert_eq!(input_ids.last_de_delayed(), None);
        input_ids.push([13, 14, 15, 16]);
        assert_eq!(input_ids.last_de_delayed(), Some([1, 6, 11, 16]));
        input_ids.push([17, 18, 19, 20]);
        assert_eq!(input_ids.last_de_delayed(), Some([5, 10, 15, 20]));
    }

    #[test]
    fn len_tracking() {
        let mut pattern = DelayPatternMaskIds::<4>::new();
        assert_eq!(pattern.len(), 0);
        pattern.push([1, 2, 3, 4]);
        assert_eq!(pattern.len(), 1);
        pattern.push([5, 6, 7, 8]);
        assert_eq!(pattern.len(), 2);
    }
}
