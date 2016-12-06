"""Implements a fully blocking kernel client.

Useful for test suites and blocking terminal interfaces.
"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

try:
    from queue import Empty  # Python 3
except ImportError:
    from Queue import Empty  # Python 2
import time

from traitlets import Type
from jupyter_client.channels import HBChannel
from jupyter_client.client import KernelClient
from .channels import ZMQSocketChannel


class BlockingKernelClient(KernelClient):
    """A BlockingKernelClient """
    
    def wait_for_ready(self, timeout=None):
        """Waits for a response when a client is blocked
        
        - Sets future time for timeout
        - Blocks on shell channel until a message is received
        - Exit if the kernel has died
        - If client times out before receiving a message from the kernel, send RuntimeError
        - Flush the IOPub channel
        """
        if timeout is None:
            abs_timeout = float('inf')
        else:
            abs_timeout = time.time() + timeout

        from ..manager import KernelManager
        if not isinstance(self.parent, KernelManager):
            # This Client was not created by a KernelManager,
            # so wait for kernel to become responsive to heartbeats
            # before checking for kernel_info reply
            while not self.is_alive():
                if time.time() > abs_timeout:
                    raise RuntimeError("Kernel didn't respond to heartbeats in %d seconds and timed out" % timeout)
                time.sleep(0.2)

        # Wait for kernel info reply on shell channel
        while True:
            try:
                msg = self.shell_channel.get_msg(block=True, timeout=1)
            except Empty:
                pass
            else:
                if msg['msg_type'] == 'kernel_info_reply':
                    self._handle_kernel_info_reply(msg)
                    break

            if not self.is_alive():
                raise RuntimeError('Kernel died before replying to kernel_info')

            # Check if current time is ready check time plus timeout
            if time.time() > abs_timeout:
                raise RuntimeError("Kernel didn't respond in %d seconds" % timeout)

        # Flush IOPub channel
        while True:
            try:
                msg = self.iopub_channel.get_msg(block=True, timeout=0.2)
            except Empty:
                break

    # The classes to use for the various channels
    shell_channel_class = Type(ZMQSocketChannel)
    iopub_channel_class = Type(ZMQSocketChannel)
    stdin_channel_class = Type(ZMQSocketChannel)
    hb_channel_class = Type(HBChannel)
