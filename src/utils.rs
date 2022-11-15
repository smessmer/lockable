use futures::{
    stream::{Stream, StreamExt},
    task::{self, Context, Poll},
};

pub trait MyStreamExt: Stream {
    /// Return the next value from the stream if there is a value ready.
    /// Otherwise, return [None].
    ///
    /// Examples
    /// -----
    /// ```ignore
    /// use lockable::utils::MyStreamExt;
    ///
    /// // A stream with a pending future that can be fulfilled later.
    /// let (sender, receiver) = futures::channel::oneshot::channel::<i32>();
    /// let mut stream = futures::stream::once(receiver);
    ///
    /// // The next value is not ready yet, so this returns None.
    /// assert_eq!(stream.next_if_ready(), None);
    ///
    /// // Fulfill the future and the next value is now ready.
    /// sender.send(42).unwrap();
    ///
    /// // The next value is now ready, so this returns Some(42).
    /// assert_eq!(Some(Ok(42)), stream.next_if_ready());
    /// ```
    fn next_if_ready(&mut self) -> Option<Self::Item>;
}

impl<T> MyStreamExt for T
where
    T: Stream + Unpin,
{
    fn next_if_ready(&mut self) -> Option<Self::Item> {
        let noop_waker = task::noop_waker();
        let mut cx = Context::from_waker(&noop_waker);
        match self.poll_next_unpin(&mut cx) {
            Poll::Ready(Some(item)) => Some(item),
            Poll::Ready(None) => None,
            Poll::Pending => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream() {
        let mut stream = futures::stream::empty::<i32>();
        assert_eq!(stream.next_if_ready(), None);
    }

    #[test]
    fn stream_with_one_item() {
        let mut stream = futures::stream::once(futures::future::ready(42));
        assert_eq!(stream.next_if_ready(), Some(42));
        assert_eq!(stream.next_if_ready(), None);
    }

    #[test]
    fn stream_with_multiple_items() {
        let mut stream = futures::stream::iter(vec![1, 2, 3]);
        assert_eq!(stream.next_if_ready(), Some(1));
        assert_eq!(stream.next_if_ready(), Some(2));
        assert_eq!(stream.next_if_ready(), Some(3));
        assert_eq!(stream.next_if_ready(), None);
    }

    #[test]
    fn stream_with_multiple_pending_items() {
        // A pending future that can be fulfilled later.
        let (sender1, receiver1) = futures::channel::oneshot::channel::<i32>();
        let (sender2, receiver2) = futures::channel::oneshot::channel::<i32>();
        let (sender3, receiver3) = futures::channel::oneshot::channel::<i32>();

        let mut stream = futures::stream::once(receiver1)
            .chain(futures::stream::once(receiver2))
            .chain(futures::stream::once(receiver3));

        assert_eq!(stream.next_if_ready(), None);

        sender1.send(1).unwrap();
        assert_eq!(stream.next_if_ready(), Some(Ok(1)));
        assert_eq!(stream.next_if_ready(), None);

        sender2.send(2).unwrap();
        assert_eq!(stream.next_if_ready(), Some(Ok(2)));
        assert_eq!(stream.next_if_ready(), None);

        sender3.send(3).unwrap();
        assert_eq!(stream.next_if_ready(), Some(Ok(3)));
        assert_eq!(stream.next_if_ready(), None);

        assert_eq!(futures::executor::block_on(stream.next()), None);
    }
}
