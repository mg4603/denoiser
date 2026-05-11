# ADR 0002: librosa for Audio Loading

**Status: Accepted**  
**Date: 11-05-2026**  


## Context

Required a method to load audio after extraction to denoise
and then to mux with video.

## Decision
Selected `librosa` as the primary library for audio loading.


## Alternatives Considered
- soundfile: rejected as it lacks built-in resampling 
- pydub: rejected due to additional dependencies and less 
  efficient array handling
- torchaudio: rejected as it introduces unnecessary complexity
  for simple loading operations
  

## Consequences
### Positive
- Consistent numpy output
- Excellent documentation and examples
- Built-in resampling and mono conversion

### Negative
- Slight performance overhead
- May be overkill for wav-only use cases
