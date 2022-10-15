use derive_more::Display;
use std::fmt::Debug;
use thiserror::Error;

/// This error is thrown by [GuardImpl::try_insert] if the entry already exists
#[derive(Error, Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum TryInsertError<V> {
    /// The entry couldn't be inserted because it already exists
    #[error("The entry couldn't be inserted because it already exists")]
    AlreadyExists {
        /// The value that was attempted to be inserted
        value: V,
    },
}

// TODO use the never type `!` instead of this once it is stabilized
/// A type that can never be instantiated. This can be used in a
/// `Result<T, Never>` to indicate that an operation cannot return
/// an error.
#[derive(Debug, Display)]
pub enum Never {}

impl std::error::Error for Never {}

/// Extension trait for `Result<T, Never>` that adds an infallible
/// version of `unwrap()`.
pub trait InfallibleUnwrap<T> {
    /// Similar to `unwrap()`, but can never fail.
    /// This is only available on `Result<T, Never>` types
    /// that are used as a result of operations that cannot
    /// return an error.
    /// Calling `infallible_unwrap()` instead of `unwrap()`
    /// uses the type system to ensure that you don't
    /// accidentally `unwrap()` a type that might contain
    /// an error.
    fn infallible_unwrap(self) -> T;
}

impl<T> InfallibleUnwrap<T> for std::result::Result<T, Never> {
    fn infallible_unwrap(self) -> T {
        // TODO Test
        self.unwrap()
    }
}
