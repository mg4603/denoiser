# ADR 0001: Use noisereduce Library for Audio Processing

**Status: Accepted**  
**Date: 11-05-2026**  

## Context

Needed audio denoising for python Video Processing tool without
GPU requirements.

## Decision

Selected `noisereduce` library for spectral subtraction-based 
noise reduction.

## Alternatives considered

- torchaudio: rejected as it requires GPU
- pedalboard: rejected as it requires for parameter tuning
- RNNnoise: rejected due to higher complexity

## Consequences

### Positive
- Lower computational overhead
- Simple integration
- CPU-only operation
- Small dependency footprint

### Negative
- Lower quality than modern ML based denoisers
- Limited effectiveness for non-stationary noise
- Slower than GPU-accelerated alternatives
