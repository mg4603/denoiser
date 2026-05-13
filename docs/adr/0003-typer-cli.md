# ADR 0003: typer for CLI

**Status: Accepted**  
**Date: 11-05-2026**  

## Context
Needed a command-line interface for `denoiser`.

## Decision
Selected `typer` as the framework for building command-line
interface.

## Alternatives Considered
- argparse: rejected due to requiring more boilerplate code
- click: rejected due to more verbose syntax
- docopt: rejected due to being less maintainable

## Consequences
### Positive
- Minimal boilerplate required
- Automatic help generation and validation
- Easy to add subcommands and options
- Clean, readable CLI definition code

### Negative
- Additional dependency beyond Python standard library
- Less flexible than building from scratch for specialized needs

