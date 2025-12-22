# JSON-RPC Contract: ACE-Step Backend Extension

**Phase 1 Output** | **Branch**: `002-ace-step` | **Date**: 2025-12-21

## Overview

This document defines the JSON-RPC 2.0 contract extensions for ACE-Step backend support. All existing methods remain backward-compatible; new fields are optional with sensible defaults.

---

## Protocol Specification

**Transport**: stdin/stdout (line-delimited JSON)
**Version**: JSON-RPC 2.0
**Encoding**: UTF-8

---

## Methods

### generate

Queues a generation request. Extended to support backend selection and ACE-Step parameters.

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "generate",
  "params": {
    "prompt": "lofi hip hop, jazzy piano, vinyl crackle",
    "duration_sec": 120,
    "backend": "ace_step",
    "seed": null,
    "priority": "normal",
    "inference_steps": 60,
    "scheduler": "euler",
    "guidance_scale": 15.0
  }
}
```

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text prompt (1-512 chars) |
| `duration_sec` | integer | Yes | - | Duration in seconds |
| `backend` | string | No | Config default | `"musicgen"` or `"ace_step"` |
| `seed` | integer\|null | No | Random | Reproducibility seed (u64) |
| `priority` | string | No | `"normal"` | `"high"` or `"normal"` |
| `inference_steps` | integer | No | 60 | ACE-Step: diffusion steps (1-200) |
| `scheduler` | string | No | `"euler"` | ACE-Step: `"euler"`, `"heun"`, `"pingpong"` |
| `guidance_scale` | number | No | 15.0 | ACE-Step: CFG scale (1.0-30.0) |

**Response** (immediate, before generation starts):
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "track_id": "a1b2c3d4e5f6...",
    "status": "Generating",
    "position": 0,
    "seed": 12345678901234,
    "backend": "ace_step"
  }
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | string | Unique identifier for this generation |
| `status` | string | `"Cached"`, `"Generating"`, or `"Queued"` |
| `position` | integer | Queue position (0 = generating now) |
| `seed` | integer | Actual seed used (returned if random) |
| `backend` | string | Backend being used |

**Errors**:

| Code | Message | When |
|------|---------|------|
| -32602 | Invalid params | JSON parsing failed |
| -32004 | Queue full | 10 requests already queued |
| -32005 | Invalid duration | Outside backend's range |
| -32006 | Invalid prompt | Empty or too long prompt |
| -32007 | Invalid backend | Unknown backend type |
| -32008 | Backend not installed | Requested backend unavailable |
| -32009 | Invalid inference steps | Steps outside 1-200 range |
| -32010 | Invalid guidance scale | Scale outside 1.0-30.0 range |
| -32011 | Invalid scheduler | Unknown scheduler type |

---

### cancel

Cancels an in-progress or queued generation.

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "cancel",
  "params": {
    "track_id": "a1b2c3d4e5f6..."
  }
}
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `track_id` | string | Yes | Track to cancel |

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "cancelled": true,
    "was_generating": true
  }
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `cancelled` | boolean | True if cancellation succeeded |
| `was_generating` | boolean | True if was actively generating (vs queued) |

**Errors**:

Note: Cancel functionality uses process termination as a fallback mechanism. These error responses are reserved for future RPC-based cancellation.

| Code | Message | When |
|------|---------|------|
| -32000 | Track not found | Unknown track_id |
| -32001 | Already complete | Track already finished |

---

### get_backends

Returns available backends and their status.

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "get_backends",
  "params": {}
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "backends": [
      {
        "type": "musicgen",
        "name": "MusicGen-Small",
        "status": "ready",
        "min_duration_sec": 5,
        "max_duration_sec": 120,
        "sample_rate": 32000,
        "model_version": "musicgen-small-fp16-v1"
      },
      {
        "type": "ace_step",
        "name": "ACE-Step-3.5B",
        "status": "not_installed",
        "min_duration_sec": 5,
        "max_duration_sec": 240,
        "sample_rate": 48000,
        "model_version": null
      }
    ],
    "default_backend": "musicgen"
  }
}
```

**Backend Status Values**:
- `"not_installed"` - Model weights not downloaded
- `"downloading"` - Download in progress
- `"loading"` - Loading into memory
- `"ready"` - Ready for generation
- `"error"` - Failed with error message

---

### download_backend

Initiates download of backend model weights.

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "download_backend",
  "params": {
    "backend": "ace_step"
  }
}
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `backend` | string | Yes | Backend to download |

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "started": true,
    "already_installed": false
  }
}
```

**Errors**:

| Code | Message | When |
|------|---------|------|
| -32007 | Invalid backend | Unknown backend type |

---

### ping

Health check (unchanged from existing).

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "ping",
  "params": {}
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "status": "ok",
    "version": "0.2.0"
  }
}
```

---

### shutdown

Graceful shutdown (unchanged from existing).

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "shutdown",
  "params": {}
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {
    "status": "shutting_down"
  }
}
```

---

## Notifications

Notifications are sent from daemon to client without a request ID.

### generation_progress

Sent during generation at 5% intervals.

```json
{
  "jsonrpc": "2.0",
  "method": "generation_progress",
  "params": {
    "track_id": "a1b2c3d4e5f6...",
    "percent": 45,
    "current_step": 27,
    "total_steps": 60,
    "eta_sec": 8.5
  }
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | string | Track being generated |
| `percent` | integer | Completion percentage (0-99) |
| `current_step` | integer | Current step/token |
| `total_steps` | integer | Total steps/tokens |
| `eta_sec` | number | Estimated seconds remaining |

---

### generation_complete

Sent when generation finishes successfully.

```json
{
  "jsonrpc": "2.0",
  "method": "generation_complete",
  "params": {
    "track_id": "a1b2c3d4e5f6...",
    "path": "/Users/user/.cache/lofi.nvim/tracks/a1b2c3d4e5f6.wav",
    "duration_sec": 120.5,
    "sample_rate": 48000,
    "generation_time_sec": 12.3,
    "backend": "ace_step",
    "model_version": "ace-step-v1-3.5b"
  }
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | string | Completed track ID |
| `path` | string | Absolute path to audio file |
| `duration_sec` | number | Actual audio duration |
| `sample_rate` | integer | Audio sample rate |
| `generation_time_sec` | number | Time taken to generate |
| `backend` | string | Backend that generated |
| `model_version` | string | Model version string |

---

### generation_error

Sent when generation fails.

```json
{
  "jsonrpc": "2.0",
  "method": "generation_error",
  "params": {
    "track_id": "a1b2c3d4e5f6...",
    "code": "MODEL_INFERENCE_FAILED",
    "message": "Numerical instability at step 42. Try a different seed."
  }
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | string | Failed track ID |
| `code` | string | Error code (see Error Codes) |
| `message` | string | Human-readable error message |

---

### generation_cancelled

Sent when generation is cancelled.

```json
{
  "jsonrpc": "2.0",
  "method": "generation_cancelled",
  "params": {
    "track_id": "a1b2c3d4e5f6...",
    "at_step": 27,
    "total_steps": 60
  }
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | string | Cancelled track ID |
| `at_step` | integer | Step when cancelled |
| `total_steps` | integer | Total steps that were planned |

---

### download_progress

Sent during model download.

```json
{
  "jsonrpc": "2.0",
  "method": "download_progress",
  "params": {
    "backend": "ace_step",
    "component": "transformer.onnx",
    "component_percent": 67,
    "overall_percent": 42,
    "bytes_downloaded": 2831155200,
    "bytes_total": 7700000000
  }
}
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `backend` | string | Backend being downloaded |
| `component` | string | Current component file |
| `component_percent` | integer | Component download progress |
| `overall_percent` | integer | Overall download progress |
| `bytes_downloaded` | integer | Total bytes downloaded |
| `bytes_total` | integer | Total bytes to download |

---

## Error Codes Summary

### Standard JSON-RPC Errors

| Code | Constant | Description |
|------|----------|-------------|
| -32700 | PARSE_ERROR | Invalid JSON |
| -32600 | INVALID_REQUEST | Invalid JSON-RPC request |
| -32601 | METHOD_NOT_FOUND | Unknown method |
| -32602 | INVALID_PARAMS | Parameter validation failed |
| -32603 | INTERNAL_ERROR | Unexpected internal error |

### Application Errors

| Code | Constant | Description |
|------|----------|-------------|
| -32000 | MODEL_NOT_FOUND | Model files not found at expected path |
| -32001 | MODEL_LOAD_FAILED | Failed to load models into memory |
| -32002 | MODEL_DOWNLOAD_FAILED | Model download failed |
| -32003 | MODEL_INFERENCE_FAILED | Inference error during generation |
| -32004 | QUEUE_FULL | Generation queue at capacity (max 10) |
| -32005 | INVALID_DURATION | Duration outside backend range |
| -32006 | INVALID_PROMPT | Prompt validation failed |
| -32007 | INVALID_BACKEND | Unknown backend type |
| -32008 | BACKEND_NOT_INSTALLED | Backend models not downloaded |
| -32009 | INVALID_INFERENCE_STEPS | Steps outside valid range (1-200) |
| -32010 | INVALID_GUIDANCE_SCALE | Scale outside valid range (1.0-30.0) |
| -32011 | INVALID_SCHEDULER | Unknown scheduler type |

---

## Backward Compatibility

All extensions maintain backward compatibility:

1. **`generate` method**: New fields (`backend`, `inference_steps`, `scheduler`, `guidance_scale`) are optional. Existing clients work unchanged.

2. **New methods**: `cancel`, `get_backends`, `download_backend` are additive.

3. **New notifications**: `generation_cancelled`, `download_progress` are additive. Clients that don't handle them can ignore.

4. **Response extensions**: New fields in responses (`backend`, `sample_rate`) are additive. Clients can ignore unknown fields.
