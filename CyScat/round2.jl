"""
round2 - Rounding to specified decimal places

Julia translation of MATLAB round2.m
Original MATLAB code by Curtis Jin (jsirius@umich.edu)
"""

"""
    round2(decimal::Int, number)

Round a number to specified decimal places.

# Arguments
- `decimal::Int`: Number of decimal places
- `number`: Number(s) to round (scalar or array)

# Returns
- Rounded value(s)
"""
function round2(decimal::Int, number)
    rounded_value = number * 10^decimal
    rounded_value = trunc.(rounded_value)
    rounded_value = rounded_value * 10^(-decimal)
    return rounded_value
end
