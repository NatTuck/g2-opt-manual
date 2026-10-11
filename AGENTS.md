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

## Broad Outline

Sections:

- Gaudi/HPU Basics
- How to optimize small programs / regions
- How to port a large language model to HPU
