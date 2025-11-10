module LoggingTools
using Printf, Logging

const ORIG_STDOUT = stdout
const ORIG_STDERR = stderr

unmute_println(xs...) = println(LoggingTools.ORIG_STDOUT, xs...)
function unmute_printf(fmt::AbstractString, args...)
    print(LoggingTools.ORIG_STDOUT, Printf.format(Printf.Format(fmt), args...))
end

macro silent(expr)
    quote
        local _val
        # Mute logging macros (@info/@warn) no matter what global_logger was set to:
        with_logger(Logging.NullLogger()) do
            # Also mute anything writing to Base.stdout/Base.stderr:
            redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    _val = $(esc(expr))
                end
            end
        end
        _val
    end
end

export @silent, unmute_println, unmute_printf
end
