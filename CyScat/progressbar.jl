"""
Progressbar
Modified by: Quan Quach on 12/12/07
Email: quan.quach@gmail.com
Original Author: Steve Hoelzer

Julia translation using ProgressMeter.jl
"""

using ProgressMeter
using Printf

# Global progress bar state
mutable struct ProgressBarState
    progress::Union{Progress, Nothing}
    start_time::Union{Float64, Nothing}
    last_fraction::Float64
end

const _global_state = ProgressBarState(nothing, nothing, 0.0)

"""
    progressbar(fraction_done=nothing, position=0)

Module-level progress bar function.

# Arguments
- `fraction_done`: Fraction of task completed (0.0 to 1.0).
  If nothing or 0, initializes/resets the progress bar
- `position`: Position parameter (unused in console version)

# Returns
- `stop_bar`: 0 for continue, 1 for stop

# Usage
```julia
progressbar(0)      # Initialize
progressbar(0.5)    # Update to 50%
progressbar(1)      # Complete
```
"""
function progressbar(fraction_done::Union{Float64, Nothing}=nothing, position::Int=0)
    if isnothing(fraction_done)
        fraction_done = 0.0
    end

    stop_bar = 0

    # Reset or initialize
    if fraction_done == 0.0
        _global_state.progress = Progress(100; showspeed=true)
        _global_state.start_time = time()
        _global_state.last_fraction = 0.0
        return stop_bar
    end

    # Update progress
    if !isnothing(_global_state.progress)
        increment = Int(floor((fraction_done - _global_state.last_fraction) * 100))
        if increment > 0
            update!(_global_state.progress, Int(floor(fraction_done * 100)))
        end
        _global_state.last_fraction = fraction_done

        # Complete
        if fraction_done >= 1.0
            finish!(_global_state.progress)
            _global_state.progress = nothing
            _global_state.start_time = nothing
            _global_state.last_fraction = 0.0
        end
    end

    return stop_bar
end

"""
    sec2timestr(sec)

Convert seconds to human-readable time string.
"""
function sec2timestr(sec::Float64)
    d = Int(floor(sec / 86400))
    sec = sec - d * 86400
    h = Int(floor(sec / 3600))
    sec = sec - h * 3600
    m = Int(floor(sec / 60))
    sec = sec - m * 60
    s = Int(floor(sec))

    if d > 0
        if d > 9
            return "$d day"
        else
            return "$d day, $h hr"
        end
    elseif h > 0
        if h > 9
            return "$h hr"
        else
            return "$h hr, $m min"
        end
    elseif m > 0
        if m > 9
            return "$m min"
        else
            return "$m min, $s sec"
        end
    else
        return "$s sec"
    end
end
