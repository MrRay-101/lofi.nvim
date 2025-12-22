--- lofi.nvim - AI Music Generation for Neovim
--- Main entry point and public API.
local M = {}

local daemon = require("lofi.daemon")
local rpc = require("lofi.rpc")
local events = require("lofi.events")

--- Re-export events module for convenience
M.events = events
M.EVENTS = events.EVENTS

--- Generation state tracking
local state = {
  generating = false,           -- True if generation is in progress
  current_track_id = nil,       -- Track ID of current generation
  pending_callbacks = {},       -- Map of track_id -> callback
  initialized = false,          -- True if setup() has been called
  default_backend = nil,        -- Default backend from config ("musicgen" or "ace_step")
  ui_cleanup = nil,             -- Function to cleanup UI on cancel
}

--- Map daemon notification methods to event names
local notification_to_event = {
  generation_progress = events.EVENTS.GENERATION_PROGRESS,
  generation_complete = events.EVENTS.GENERATION_COMPLETE,
  generation_error = events.EVENTS.GENERATION_ERROR,
  download_progress = events.EVENTS.DOWNLOAD_PROGRESS,
}

--- Handle notifications from daemon
--- @param method string notification method name
--- @param params table notification parameters
local function handle_notification(method, params)
  local event = notification_to_event[method]
  if not event then
    return
  end

  -- Handle generation lifecycle
  local track_id = params.track_id

  if method == "generation_complete" then
    state.generating = false
    state.current_track_id = nil

    -- Call pending callback for this track
    local callback = state.pending_callbacks[track_id]
    if callback then
      state.pending_callbacks[track_id] = nil
      vim.schedule(function()
        callback(nil, params)
      end)
    end
  elseif method == "generation_error" then
    state.generating = false
    state.current_track_id = nil

    -- Call pending callback with error
    local callback = state.pending_callbacks[track_id]
    if callback then
      state.pending_callbacks[track_id] = nil
      vim.schedule(function()
        callback({ code = params.code, message = params.message }, nil)
      end)
    end
  end

  -- Emit event to all listeners
  events.emit(event, params)
end

--- Initialize lofi plugin with configuration
--- @param opts table|nil configuration options
---   - daemon_path: string|nil - Path to lofi-daemon binary
---   - model_path: string|nil - Path to ONNX model directory
---   - device: string|nil - Device selection: "auto", "cpu", "cuda", "metal"
---   - threads: number|nil - CPU thread count (nil = auto)
---   - backend: string|nil - Default backend: "musicgen" or "ace_step"
function M.setup(opts)
  opts = opts or {}
  daemon.setup(opts)
  state.initialized = true
  state.default_backend = opts.backend
end

--- Check if generation is currently in progress
--- @return boolean true if generating
function M.is_generating()
  return state.generating
end

--- Get the current track ID being generated
--- @return string|nil track ID or nil if not generating
function M.current_track()
  return state.current_track_id
end

--- Generate music from a text prompt
--- @param opts string|table prompt string or generation options table
---   - prompt: string - Text description of desired music (required)
---   - duration_sec: number|nil - Duration in seconds (5-120 for MusicGen, 5-240 for ACE-Step, default 30)
---   - seed: number|nil - Random seed for reproducibility (nil = random)
---   - priority: string|nil - "normal" or "high" (default "normal")
---   - backend: string|nil - Backend to use: "musicgen" or "ace_step" (default from config)
---   - inference_steps: number|nil - ACE-Step only: diffusion steps (1-200, default 60)
---   - scheduler: string|nil - ACE-Step only: "euler", "heun", or "pingpong" (default "euler")
---   - guidance_scale: number|nil - ACE-Step only: CFG scale (1.0-30.0, default 15.0)
--- @param callback function|nil callback receiving (error, result)
---   - error: table|nil - { code, message } on failure
---   - result: table|nil - { track_id, path, duration_sec, backend, ... } on success
--- @return boolean success true if request was sent
function M.generate(opts, callback)
  if not state.initialized then
    M.setup({})
  end

  -- Accept string as shorthand for {prompt = string}
  if type(opts) == "string" then
    opts = { prompt = opts }
  end

  -- Validate options
  if not opts or not opts.prompt or opts.prompt == "" then
    if callback then
      vim.schedule(function()
        callback({ code = -32006, message = "Prompt is required" }, nil)
      end)
    end
    return false
  end

  -- Determine backend (from opts, or state default, or nil for daemon default)
  local backend = opts.backend or state.default_backend

  -- Validate duration if provided (backend-specific limits checked by daemon)
  if opts.duration_sec then
    local max_duration = backend == "ace_step" and 240 or 120
    if opts.duration_sec < 5 or opts.duration_sec > max_duration then
      if callback then
        vim.schedule(function()
          callback({
            code = -32005,
            message = string.format("Duration must be between 5 and %d seconds", max_duration)
          }, nil)
        end)
      end
      return false
    end
  end

  -- Validate priority if provided
  if opts.priority and opts.priority ~= "normal" and opts.priority ~= "high" then
    if callback then
      vim.schedule(function()
        callback({ code = -32602, message = "Priority must be 'normal' or 'high'" }, nil)
      end)
    end
    return false
  end

  -- Validate backend if provided
  if backend and backend ~= "musicgen" and backend ~= "ace_step" then
    if callback then
      vim.schedule(function()
        callback({ code = -32007, message = "Backend must be 'musicgen' or 'ace_step'" }, nil)
      end)
    end
    return false
  end

  -- Initialize RPC if needed (starts daemon)
  if not rpc.init(handle_notification) then
    if callback then
      vim.schedule(function()
        callback({ code = -32000, message = "Failed to start daemon" }, nil)
      end)
    end
    return false
  end

  -- Build request params
  local params = {
    prompt = opts.prompt,
    duration_sec = opts.duration_sec or 30,
    seed = opts.seed,
    priority = opts.priority or "normal",
    backend = backend,
    -- ACE-Step specific parameters
    inference_steps = opts.inference_steps,
    scheduler = opts.scheduler,
    guidance_scale = opts.guidance_scale,
  }

  -- Send generate request
  local request_id = rpc.send_request("generate", params, function(err, result)
    if err then
      if callback then
        callback(err, nil)
      end
      return
    end

    -- Store track ID and callback for completion notification
    local track_id = result.track_id
    state.generating = true
    state.current_track_id = track_id

    if callback then
      state.pending_callbacks[track_id] = callback
    end

    -- Emit generation_start event
    events.emit(events.EVENTS.GENERATION_START, {
      track_id = track_id,
      prompt = opts.prompt,
      duration_sec = params.duration_sec,
      seed = result.seed,
      position = result.position,
      backend = result.backend,
    })
  end)

  return request_id ~= nil
end

--- Check if daemon is running
--- @return boolean true if daemon process is active
function M.is_daemon_running()
  return daemon.is_running()
end

--- Get available backends and their status
--- @param callback function callback receiving (error, result)
---   - error: table|nil - { code, message } on failure
---   - result: table|nil - { backends: array, default_backend: string }
---     Each backend has: { type, name, status, min_duration_sec, max_duration_sec, sample_rate, model_version? }
--- @return boolean success true if request was sent
function M.get_backends(callback)
  if not state.initialized then
    M.setup({})
  end

  -- Initialize RPC if needed (starts daemon)
  if not rpc.init(handle_notification) then
    if callback then
      vim.schedule(function()
        callback({ code = -32000, message = "Failed to start daemon" }, nil)
      end)
    end
    return false
  end

  local request_id = rpc.send_request("get_backends", {}, function(err, result)
    if callback then
      callback(err, result)
    end
  end)

  return request_id ~= nil
end

--- Stop the daemon gracefully
function M.stop()
  rpc.shutdown(false)
  state.generating = false
  state.current_track_id = nil
  state.pending_callbacks = {}
end

--- Force stop the daemon
function M.force_stop()
  rpc.shutdown(true)
  state.generating = false
  state.current_track_id = nil
  state.pending_callbacks = {}
end

--- Cancel in-progress generation by killing the daemon
function M.cancel()
  -- Kill daemon process
  if vim.fn.has("win32") == 1 then
    vim.fn.system("taskkill /F /IM lofi-daemon.exe")
  else
    vim.fn.system("pkill -9 -f lofi-daemon")
  end
  -- Cleanup UI
  if state.ui_cleanup then
    state.ui_cleanup()
    state.ui_cleanup = nil
  end
  state.generating = false
  state.current_track_id = nil
  state.pending_callbacks = {}
end

--- Register an event handler (convenience wrapper)
--- @param event string event name from M.EVENTS
--- @param callback function handler receiving event data
--- @return function unsubscribe function
function M.on(event, callback)
  return events.on(event, callback)
end

-- Helper to run generation with UI
local function run_generation(prompt, duration, backend)
  -- Create floating window for progress
  local buf = vim.api.nvim_create_buf(false, true)
  local width = 50
  local win = vim.api.nvim_open_win(buf, false, {
    relative = "editor",
    width = width,
    height = 1,
    row = 1,
    col = vim.o.columns - width - 2,
    style = "minimal",
    border = "rounded",
  })

  local function update(msg)
    if vim.api.nvim_buf_is_valid(buf) then
      vim.api.nvim_buf_set_lines(buf, 0, -1, false, { msg })
    end
  end

  local function close()
    if vim.api.nvim_win_is_valid(win) then
      vim.api.nvim_win_close(win, true)
    end
  end

  local backend_label = backend and (" [" .. backend .. "]") or ""
  update("[lofi]" .. backend_label .. " Generating: " .. prompt)

  local unsub_progress, unsub_complete, unsub_error

  local function cleanup()
    if unsub_progress then unsub_progress() end
    if unsub_complete then unsub_complete() end
    if unsub_error then unsub_error() end
    close()
    state.ui_cleanup = nil
  end

  -- Store cleanup function so cancel can call it
  state.ui_cleanup = cleanup

  unsub_progress = M.on("generation_progress", function(data)
    update("[lofi] " .. data.percent .. "% - " .. prompt)
  end)

  unsub_complete = M.on("generation_complete", function(data)
    cleanup()
    M.last_track = data.path
    vim.notify("[lofi] Done! :LofiPlay to play (" .. (data.backend or "unknown") .. ")", vim.log.levels.INFO)
    -- Auto-play
    vim.fn.jobstart({ "afplay", data.path })
  end)

  unsub_error = M.on("generation_error", function(data)
    cleanup()
    vim.notify("[lofi] Error: " .. data.message, vim.log.levels.ERROR)
  end)

  M.generate({ prompt = prompt, duration_sec = duration, backend = backend })
end

-- Create :Lofi command on module load
vim.api.nvim_create_user_command("Lofi", function(cmd)
  local args = cmd.args
  if args == "" then
    vim.notify("[lofi] Usage: :Lofi <prompt> [duration]", vim.log.levels.ERROR)
    return
  end

  -- Check if last word is a number (duration)
  local prompt, duration = args, 10
  local last_word = args:match("(%d+)$")
  if last_word then
    duration = tonumber(last_word) or 10
    prompt = args:sub(1, -(#last_word + 2)) -- remove duration and space
    if prompt == "" then
      prompt = args
      duration = 10
    end
  end

  run_generation(prompt, duration, nil)
end, { nargs = "*", desc = "Generate lofi music from prompt" })

-- Create :LofiAce command for ACE-Step backend
vim.api.nvim_create_user_command("LofiAce", function(cmd)
  local args = cmd.args
  if args == "" then
    vim.notify("[lofi] Usage: :LofiAce <prompt> [duration]", vim.log.levels.ERROR)
    return
  end

  -- Check if last word is a number (duration)
  local prompt, duration = args, 30
  local last_word = args:match("(%d+)$")
  if last_word then
    duration = tonumber(last_word) or 30
    prompt = args:sub(1, -(#last_word + 2))
    if prompt == "" then
      prompt = args
      duration = 30
    end
  end

  run_generation(prompt, duration, "ace_step")
end, { nargs = "*", desc = "Generate music with ACE-Step backend" })

-- Create :LofiBackends command to show available backends
vim.api.nvim_create_user_command("LofiBackends", function()
  M.get_backends(function(err, result)
    if err then
      vim.notify("[lofi] Error: " .. (err.message or "unknown"), vim.log.levels.ERROR)
      return
    end
    vim.schedule(function()
      local lines = { "Available backends (default: " .. result.default_backend .. "):" }
      for _, b in ipairs(result.backends) do
        local status_icon = b.status == "ready" and "✓" or "✗"
        table.insert(lines, string.format("  %s %s (%s) - %d-%ds @ %dHz",
          status_icon, b.name, b.type, b.min_duration_sec, b.max_duration_sec, b.sample_rate))
      end
      vim.notify(table.concat(lines, "\n"), vim.log.levels.INFO)
    end)
  end)
end, { desc = "Show available lofi backends" })

-- :LofiPlay command
vim.api.nvim_create_user_command("LofiPlay", function()
  if M.last_track then
    vim.fn.jobstart({ "afplay", M.last_track })
  else
    vim.notify("[lofi] No track to play", vim.log.levels.WARN)
  end
end, { desc = "Play last generated track" })

-- :LofiStop command
vim.api.nvim_create_user_command("LofiStop", function()
  vim.fn.jobstart({ "pkill", "-f", "afplay" })
end, { desc = "Stop playing" })

-- :LofiCancel command
vim.api.nvim_create_user_command("LofiCancel", function()
  M.cancel()
  vim.notify("[lofi] Generation cancelled", vim.log.levels.INFO)
end, { desc = "Cancel in-progress generation" })

-- Set global for convenience
_G.lofi = M

return M
