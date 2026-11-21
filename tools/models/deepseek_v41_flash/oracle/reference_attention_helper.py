#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runs the reference's sparse attention kernel, unchanged, on another card, for reference_oracle.py.

Why it exists. With the model's sizes (64 heads of 512, key tiles of 64) the reference's kernel asks for 141,312
bytes of shared memory per block (kernel.py: sparse_attn_kernel). An RTX PRO 6000 allows 101,376, so the kernel
cannot start there ("Failed to set the allowed dynamic shared memory size"); a CMP 170HX allows 166,912 and has the
16-bit tensor-core products the kernel uses. The rest of the reference needs the 8-bit float kernels and stays on
the first card. So this one call crosses: the oracle hands its five inputs to a child process that sees only the
other card, the child calls the reference's own sparse_attn there, and the output comes back. Nothing of the
arithmetic is rewritten; what differs from a run on one card is which card's tensor cores add the products.

    reference_attention_helper.py --serve <checkpoint>/inference     (started by install(), not by hand)

Exchange: the parent saves the inputs with torch.save under /dev/shm and writes that path as one line to the
child's standard input; the child answers "ok" after saving the output beside it, or "error <text>". /dev/shm is
open to every user, and the login manager removes a user's files there when that user's last login session ends
(RemoveIPC, its default), which an ssh command elsewhere can do at any moment; then anyone may make a folder or link
of the same name. So both sides open the folder without following a link and use it only where it is a directory of
this user's that no other user can enter, each file through that descriptor, never following a link, the inputs and
the output made new (never written over a file there); before each call the parent makes a fresh folder (mkdtemp: a
new random name, 0700) where its folder is gone or is not that, and a child that finds it so answers an error.
Measurement tool: not part of the engine.
"""
import os
import stat
import subprocess
import sys
import tempfile

import torch

SHARED_MEMORY_FOLDER = "/dev/shm"
ANSWER_OK = "ok"
ANSWER_ERROR = "error"
OUTPUT_SUFFIX = ".output"
SERVE_FLAG = "--serve"
REPEAT_CHECK_VARIABLE = "REFERENCE_ATTENTION_REPEAT_CHECK"    # set: every call is made twice and compared
EXPECTED_ARGUMENT_COUNT = 3
FOLDER_PREFIX = "reference_attention_"
PRIVATE_FOLDER_MODE, PRIVATE_FILE_MODE = 0o700, 0o600       # mkdtemp's mode; the files' own
FOLDER_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW


def private_folder(path):
    """A descriptor of the folder at path, never through a link. Raises OSError where it is gone or not a folder,
    PermissionError where it is not this user's alone (the module text)."""
    descriptor = os.open(path, FOLDER_FLAGS)
    status = os.fstat(descriptor)
    if status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != PRIVATE_FOLDER_MODE:
        os.close(descriptor)
        raise PermissionError(f"{path} is not a folder of this user's alone")
    return descriptor


def opened(folder, name, flags):
    """A file object of `name` in the folder of descriptor `folder`, never through a link: flags NEW_FILE_FLAGS to
    write a new file, os.O_RDONLY to read one."""
    return os.fdopen(os.open(name, flags | os.O_NOFOLLOW, PRIVATE_FILE_MODE, dir_fd = folder),
                     "rb" if flags == os.O_RDONLY else "wb")


def serve(inference_folder):
    """The child: the reference's kernel module, imported from the checkpoint, on the one card it can see."""
    answers = sys.stdout                        # the kernel compiler prints: keep its text off the answer stream
    sys.stdout = sys.stderr
    sys.path.insert(0, inference_folder)
    import kernel                               # the reference's kernel.py
    calls_checked, calls_that_differed = 0, 0
    for line in sys.stdin:
        path, folder = line.rstrip("\n"), None
        try:
            folder = private_folder(os.path.dirname(path))
            with opened(folder, os.path.basename(path), os.O_RDONLY) as handle:
                inputs = torch.load(handle)
            with torch.inference_mode():
                output = kernel.sparse_attn(inputs["q"].cuda(), inputs["kv"].cuda(), inputs["attn_sink"].cuda(),
                                            inputs["topk_idxs"].cuda(), inputs["softmax_scale"]).cpu()
                if os.environ.get(REPEAT_CHECK_VARIABLE):       # the same call once more: the card must repeat itself
                    again = kernel.sparse_attn(inputs["q"].cuda(), inputs["kv"].cuda(), inputs["attn_sink"].cuda(),
                                               inputs["topk_idxs"].cuda(), inputs["softmax_scale"]).cpu()
                    calls_checked += 1
                    calls_that_differed += int(not torch.equal(output.view(torch.int16), again.view(torch.int16)))
                    print(f"attention helper: {calls_that_differed} of {calls_checked} calls differed when repeated",
                          file = sys.stderr, flush = True)
            with opened(folder, os.path.basename(path) + OUTPUT_SUFFIX, NEW_FILE_FLAGS) as handle:
                torch.save(output, handle)
            answer = ANSWER_OK
        except Exception as failure:            # the parent decides what a failure means
            answer = f"{ANSWER_ERROR} {type(failure).__name__}: {' '.join(str(failure).split())}"
        finally:
            if folder is not None:
                os.close(folder)
        print(answer, file = answers, flush = True)


class AttentionHelper:
    """The parent's side: a child bound to another card, and the function that stands in for kernel.sparse_attn."""

    def __init__(self, checkpoint_directory, visible_card):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES = str(visible_card))
        self.folder = tempfile.mkdtemp(prefix = FOLDER_PREFIX, dir = SHARED_MEMORY_FOLDER)
        self.calls = 0
        self.child = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), SERVE_FLAG, os.path.join(checkpoint_directory, "inference")],
            stdin = subprocess.PIPE, stdout = subprocess.PIPE, text = True, env = environment)

    def sparse_attn(self, q, kv, attn_sink, topk_idxs, softmax_scale):
        try:
            folder = private_folder(self.folder)
        except OSError:                                                    # gone, or not ours alone: a fresh one
            self.folder = tempfile.mkdtemp(prefix = FOLDER_PREFIX, dir = SHARED_MEMORY_FOLDER)
            folder = private_folder(self.folder)
        name = f"call_{self.calls}"
        self.calls += 1
        try:                                                               # every file of the call through `folder`
            with opened(folder, name, NEW_FILE_FLAGS) as handle:
                torch.save({"q": q.cpu(), "kv": kv.cpu(), "attn_sink": attn_sink.cpu(),
                            "topk_idxs": topk_idxs.cpu(), "softmax_scale": softmax_scale}, handle)
            self.child.stdin.write(os.path.join(self.folder, name) + "\n")
            self.child.stdin.flush()
            answer = self.child.stdout.readline().rstrip("\n")
            if answer != ANSWER_OK:
                raise RuntimeError(f"the attention helper answered {answer!r} (an empty answer: it died)")
            with opened(folder, name + OUTPUT_SUFFIX, os.O_RDONLY) as handle:
                output = torch.load(handle).to(q.device)
            for each in (name, name + OUTPUT_SUFFIX):
                os.remove(each, dir_fd = folder)
        finally:
            os.close(folder)
        if output.shape != q.shape or output.dtype != q.dtype:
            raise RuntimeError("the attention helper returned another shape or type than the query's")
        return output

    def close(self):
        self.child.stdin.close()
        self.child.wait()
        if os.path.isdir(self.folder):
            os.rmdir(self.folder)


def install(reference_model, checkpoint_directory, visible_card):
    """Makes the reference model module call the helper wherever it called sparse_attn. Returns the helper."""
    if not hasattr(reference_model, "sparse_attn"):
        raise RuntimeError("the reference model module has no sparse_attn to stand in for")
    helper = AttentionHelper(checkpoint_directory, visible_card)
    reference_model.sparse_attn = helper.sparse_attn
    return helper


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT or sys.argv[1] != SERVE_FLAG:
        sys.exit(__doc__)
    serve(sys.argv[2])
