/// Extension trait for `Result<T, !>` that adds [infallible_unwrap()](InfallibleUnwrap::infallible_unwrap),
/// an infallible version of [unwrap()](Result::unwrap).
pub trait InfallibleUnwrap<T> {
    /// Similar to [unwrap()](Result::unwrap), but can never fail.
    /// This is only available on [Result<T, !>] types
    /// that are used as a result of operations that cannot
    /// return an error.
    /// Calling [infallible_unwrap()](InfallibleUnwrap::infallible_unwrap) instead of [unwrap()](Result::unwrap)
    /// uses the type system to ensure that you don't
    /// accidentally unwrap a type that might contain
    /// an error.
    fn infallible_unwrap(self) -> T;
}

impl<T> InfallibleUnwrap<T> for std::result::Result<T, !> {
    fn infallible_unwrap(self) -> T {
        self.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infallible_unwrap() {
        let result: Result<i64, !> = Ok(4);
        assert_eq!(4, result.infallible_unwrap());

        let result: Result<(), !> = Ok(());
        {
            #![allow(clippy::unit_cmp)]
            assert_eq!((), result.infallible_unwrap());
        }
    }
}
