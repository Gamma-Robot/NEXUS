from collections.abc import Sequence

import torch
from mjlab.utils.buffers import CircularBuffer


class AsyncCircularBuffer(CircularBuffer):
    def __init__(self, max_len: int, batch_size: int, device: str):
        super().__init__(max_len, batch_size, device)
        # Keep per-batch pointers for asynchronous writes.
        self._pointer = -torch.ones(self._batch_size, dtype=torch.long, device=self._device)

    @property
    def buffer(self) -> torch.Tensor:
        if torch.any(self._num_pushes == 0):
            raise RuntimeError("Attempting to access a buffer that is not fully initialized.")
        return self.get_by_batch_ids()

    def get_by_batch_ids(self, batch_ids: Sequence[int] | None = None) -> torch.Tensor:
        # Index seems too large, potentially needing speed optimization. But we may wait and see.
        if batch_ids is None:
            batch_ids = self._all_indices
            selected_buf = self._buffer
            selected_batch_size = self._batch_size
        else:
            batch_ids = torch.as_tensor(batch_ids, device=self._device, dtype=torch.long)
            selected_buf = self._buffer[:, batch_ids, ...]
            selected_batch_size = batch_ids.size(0)

        shifts = self.max_length - self._pointer - 1
        selected_shifts = shifts[batch_ids]
        T = self.max_length
        arange = torch.arange(T, device=self._device)  # (T,)
        index = (
            (arange[:, None] - selected_shifts[None, :]) % T
        ).long()  # (T, 1) - (1, selected_B) -> (T, selected_B)
        extra_shape = selected_buf.shape[2:]  # (*D)
        index = index.view(
            T, selected_batch_size, *([1] * len(extra_shape))
        )  # (T, selected_B, 1....)
        index = index.expand(T, selected_batch_size, *extra_shape)  # (T, selected_B, *D)
        buf = torch.gather(selected_buf, dim=0, index=index)
        return torch.transpose(buf, dim0=0, dim1=1)

    def append(self, data: torch.Tensor, batch_ids: Sequence[int] | None = None):
        if batch_ids is None:
            if data.shape[0] != self._batch_size:
                raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}.")
            batch_ids = self._all_indices
        else:
            batch_ids = torch.as_tensor(batch_ids, device=self._device, dtype=torch.long)
            if data.shape[0] != len(batch_ids):
                raise ValueError(
                    f"Data shape {data.shape[0]} does not match batch_ids length {len(batch_ids)}."
                )

        data = data.to(self._device)

        if self._buffer is None:
            self._pointer = -torch.ones(self._batch_size, dtype=torch.long, device=self._device)
            self._buffer = torch.empty(
                (self.max_length, self._batch_size) + data.shape[1:],
                device=self._device,
                dtype=data.dtype,
            )

        self._pointer[batch_ids] = (self._pointer[batch_ids] + 1) % self.max_length
        self._buffer[self._pointer[batch_ids], batch_ids] = data
        is_first_push = self._num_pushes[batch_ids] == 0

        if torch.any(is_first_push):
            batch_ids = torch.as_tensor(batch_ids, device=self._device)
            first_push_batch_ids = batch_ids[is_first_push]
            self._buffer[:, first_push_batch_ids] = data[is_first_push]

        self._num_pushes[batch_ids] += 1

    def __getitem__(
        self, key: torch.Tensor | None = None, batch_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if batch_ids is None:
            batch_ids = self._all_indices
        else:
            batch_ids = torch.as_tensor(batch_ids, device=self._device, dtype=torch.long)

        if key is None:
            return self.get_by_batch_ids(batch_ids)

        if isinstance(key, int):
            key = torch.full((len(batch_ids),), key, dtype=torch.long, device=self._device)
        else:
            key = key.to(device=self._device, dtype=torch.long)
            if key.ndim == 0:
                key = key.expand(len(batch_ids))
            if key.numel() != len(batch_ids):
                raise ValueError(
                    f"Batch IDs length {len(batch_ids)} does not match key shape {key.shape[0]}."
                )

        if torch.any(self._num_pushes[batch_ids] == 0) or self._buffer is None:
            raise RuntimeError(
                "Attempting to retrieve data on an empty circular buffer. Please append data first."
            )

        current_pointers = self._pointer[batch_ids]
        valid_keys = torch.minimum(key, self._num_pushes[batch_ids] - 1)
        index_in_buffer = torch.remainder(current_pointers - valid_keys, self.max_length)
        return self._buffer[index_in_buffer, batch_ids]
