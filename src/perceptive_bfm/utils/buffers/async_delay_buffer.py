from collections.abc import Sequence

import torch
from mjlab.utils.buffers import DelayBuffer

from .async_circular_buffer import AsyncCircularBuffer


class AsyncDelayBuffer(DelayBuffer):
    """Asynchronous delay buffer that allows retrieving stored data with delays asynchronously for each batch index."""

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the asynchronous delay buffer.

        Args:
            history_length: The history of the buffer, i.e., the number of time steps in the past that the data
                will be buffered. It is recommended to set this value equal to the maximum time-step lag that
                is expected. The minimum acceptable value is zero, which means only the latest data is stored.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
        """
        super().__init__(
            min_lag=0, max_lag=max(0, history_length), batch_size=batch_size, device=device
        )
        self._history_length = max(0, history_length)
        self._circular_buffer = AsyncCircularBuffer(self._history_length + 1, batch_size, device)
        self._min_time_lag = 0
        self._max_time_lag = 0
        self._time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)

    @property
    def history_length(self) -> int:
        """The history length of the delay buffer.

        If zero, only the latest data is stored. If one, the latest and the previous data are stored, and so on.
        """
        return self._history_length

    @property
    def min_time_lag(self) -> int:
        """Minimum amount of time steps that can be delayed.

        This value cannot be negative or larger than :attr:`max_time_lag`.
        """
        return self._min_time_lag

    @property
    def max_time_lag(self) -> int:
        """Maximum amount of time steps that can be delayed.

        This value cannot be greater than :attr:`history_length`.
        """
        return self._max_time_lag

    @property
    def time_lags(self) -> torch.Tensor:
        """The time lag across each batch index.

        The shape of the tensor is (batch_size, ). The value at each index represents the delay for that index.
        This value is used to retrieve the data from the buffer.
        """
        return self._time_lags

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids: Sequence[int] | None = None):
        """Sets the time lag for the delay buffer across the provided batch indices.

        Args:
            time_lag: The desired delay for the buffer.

              * If an integer is provided, the same delay is set for the provided batch indices.
              * If a tensor is provided, the delay is set for each batch index separately. The shape of the tensor
                should be (len(batch_ids),).

            batch_ids: The batch indices for which the time lag is set. Default is None, which sets the time lag
                for all batch indices.

        Raises:
            TypeError: If the type of the :attr:`time_lag` is not int or integer tensor.
            ValueError: If the minimum time lag is negative or the maximum time lag is larger than the history length.
        """
        # resolve batch indices
        if batch_ids is None:
            batch_ids = slice(None)

        # parse requested time_lag
        if isinstance(time_lag, int):
            # set the time lags across provided batch indices
            self._time_lags[batch_ids] = time_lag
        elif isinstance(time_lag, torch.Tensor):
            # check valid dtype for time_lag: must be int or long
            if time_lag.dtype not in [torch.int, torch.long]:
                raise TypeError(
                    f"Invalid dtype for time_lag: {time_lag.dtype}. Expected torch.int or torch.long."
                )
            # set the time lags
            self._time_lags[batch_ids] = time_lag.to(device=self.device)
        else:
            raise TypeError(
                f"Invalid type for time_lag: {type(time_lag)}. Expected int or integer tensor."
            )

        # compute the min and max time lag
        self._min_time_lag = int(torch.min(self._time_lags).item())
        self._max_time_lag = int(torch.max(self._time_lags).item())
        # check that time_lag is feasible
        if self._min_time_lag < 0:
            raise ValueError(
                f"The minimum time lag cannot be negative. Received: {self._min_time_lag}"
            )
        if self._max_time_lag > self._history_length:
            raise ValueError(
                f"The maximum time lag cannot be larger than the history length. Received: {self._max_time_lag}"
            )

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the data in the delay buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        self._circular_buffer.reset(batch_ids)

    def compute(self, data: torch.Tensor, batch_ids: Sequence[int] | None = None) -> torch.Tensor:
        if batch_ids is None:
            # add the new data to the last layer
            self._circular_buffer.append(data)
            # return output
            delayed_data = self._circular_buffer[self._time_lags]
            return delayed_data.clone()
        else:
            if len(batch_ids) != data.shape[0]:
                raise ValueError(
                    f"Batch IDs length {len(batch_ids)} does not match data shape {data.shape[0]}."
                )

        # add the new data to the last layer
        self._circular_buffer.append(data, batch_ids)
        # return the output
        delayed_data = self._circular_buffer.__getitem__(self._time_lags[batch_ids], batch_ids)
        return delayed_data.clone()
