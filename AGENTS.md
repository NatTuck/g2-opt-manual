# Gaudi 2 HPU Optimization Manual

We're building a clear document for how to write code that will
execute on Gaudi 2 HPUs and run fast.

For now, this document is made up of markdown files (organized into chapters)
and example code (python, C++, TPC, etc, organized by chapter).

The markdown text should refernce the code examples for both how to do stuff and
as demonstrations / evidence of *why* specific approaches and performance models
are appropriate.

This document should be sufficient for a human or LLM to get up to speed on
how to program for Gaudi 2.

There's no reason for large data files in this repo. Probably 100% code and readable text.

## TLDR File

As we update the rest of the manual, we're going to maintain `00-tldr.md` as a token 
efficent summary of the key ideas. This file must always include:

- A basic set of common terms and definions: lazy vs. eager vs. compiled, graphs vs. recipes,
etc.
- Strong and clear about the device / compiler programming model, how specialization works.
- Strong statements about avoiding obvious slowdowns:
  - All HPU code must be graph compiled.
  - Tensor data must stay on the HPU once it's there, absolutely no device to host transfers
    mid computation, not even a little bit.
  - Graph breaks and syncs are the big cost to avoid.

As changes are made to the manual, the tldr file should be kept up to date. It should stay as compact as it reasonably
can be, never more than 10,000 tokens.
