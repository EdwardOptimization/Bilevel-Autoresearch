# Experimental Branch Status

This branch preserves the complete research contribution from
[PR #2](https://github.com/EdwardOptimization/Bilevel-Autoresearch/pull/2) by
[@llvm-x86](https://github.com/llvm-x86), including its original commit history
and experimental artifacts. We are grateful for the substantial implementation,
testing, and documentation effort behind it.

## Purpose

The branch is an incubation and research-record branch for the optional
tri-level extension. Keeping it separate from `main` allows the ideas and
evidence to remain available without presenting the extension as part of the
currently supported package.

## Current interpretation

The present experiment should be read as **inconclusive** rather than as a
confirmatory positive or negative efficacy result. The Level-3 treatment was not
applied reliably, several matched runs did not complete, and the remaining
successful pairs were below the experiment's preregistered reporting threshold.

This interpretation does not diminish the value of the engineering exploration.
In particular, the branch contains potentially reusable work around mechanism
tabu tracking, adaptive scheduling, validation, and Level-3 orchestration.

## Path toward `main`

Reusable components are welcome as focused, code-only follow-up pull requests
with dedicated tests and CI coverage. Experimental run artifacts and research
reports can remain on this branch, or be published separately as versioned
research artifacts.

Status recorded by the maintainers on 2026-07-23.
