# ADR 0004: moviepy for Video Muxing

**Status: Accepted**  
**Date: 12-05-2026**  

## Context
Needed to combine video and cleaned audio (after denoising)
while ensuring high quality output with minimal performance
overhead.

## Decision
Adopt `moviepy` for muxing operations. It offers reliable
audio-video synchronization, with minimal configuration 
requirement.

## Alternatives Considered
- FFmpeg-python: rejected due to more complex API
- OpenCV: rejected due to limited audio handling capabilities

## Consequences
### Positive
- Simple python interface
- Good format support
- Integrates with existing codebase

### Negative
- Additional dependency
- May have performance overhead for large files
- `moviepy` wraps `ffmpeg` internally
