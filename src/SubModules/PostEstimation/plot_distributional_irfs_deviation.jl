"""
    plot_distributional_irfs_deviation(
        shocks_to_plot::Vector{Tuple{Symbol,String}},
        vars_to_plot::Vector{Tuple{String,String}},
        IRFs_to_plot::Dict{String,Array{Float64}},
        IRFs_order::Vector{Symbol},
        n_par;
        bounds::Dict{String,Tuple{Float64,Float64}} = Dict{String,Tuple{Float64,Float64}}(),
        horizon::Int64 = 40,
        legend::Bool = false,
        show_fig::Bool = true,
        save_fig::Bool = false,
        path::String = "",
        fps::Int = 2,
        suffix::String = ""
    )

Plots the deviation of distributional impulse response functions (IRFs) from their 
steady state for specified shocks and variables. 
"""
function plot_distributional_irfs_deviation(
    shocks_to_plot::Vector{Tuple{Symbol,String}},
    vars_to_plot::Vector{Tuple{String,String}},
    IRFs_to_plot::Dict{String,Array{Float64}},
    IRFs_order::Vector{Symbol},
    n_par;
    bounds::Dict{String,Tuple{Float64,Float64}} = Dict{String,Tuple{Float64,Float64}}(),
    horizon::Int64 = 40,
    legend::Bool = false,
    show_fig::Bool = true,
    save_fig::Bool = false,
    path::String = "",
    fps::Int = 2,
    suffix::String = "",
)

    # Remove variables from vars_to_plot that are not in IRFs_to_plot
    filtered_vars_to_plot = []
    for (var, lab) in vars_to_plot
        if var in keys(IRFs_to_plot)
            push!(filtered_vars_to_plot, (var, lab))
        else
            @warn "The variable $var not found in IRFs_to_plot, removed from vars_to_plot."
        end
    end
    vars_to_plot = filtered_vars_to_plot

    # Remove variables from vars_to_plot that have unsupported number of dimensions
    filtered_vars_to_plot = []
    for (var, lab) in vars_to_plot
        if ndims(IRFs_to_plot[var]) in (3, 4)
            push!(filtered_vars_to_plot, (var, lab))
        else
            @warn "The variable $var has unsupported number of dimensions $(ndims(IRFs_to_plot[var])), removed from vars_to_plot."
        end
    end
    vars_to_plot = filtered_vars_to_plot

    # Unpack variables (fields) and labels
    vars = [vars_to_plot[i][1] for i in eachindex(vars_to_plot)]
    labs = [vars_to_plot[i][2] for i in eachindex(vars_to_plot)]

    # Remove shocks from shocks_to_plot that are not in IRFs_order
    filtered_shocks_to_plot = []
    for (i_shock, i_shock_lab) in shocks_to_plot
        if i_shock in IRFs_order
            push!(filtered_shocks_to_plot, (i_shock, i_shock_lab))
        else
            @warn "The shock $i_shock not found in IRFs_order, removed from shocks_to_plot."
        end
    end
    shocks_to_plot = filtered_shocks_to_plot

    # Create plots for each shock
    for (i_shock, i_shock_lab) in shocks_to_plot

        # Find position of current shock (i_shock) in IRFs array (IRFs_order)
        idx = findfirst(x -> x == i_shock, IRFs_order)

        # Create a plot for each variable
        for (i, (var, lab)) in enumerate(zip(vars, labs))
            # Select IRFs for the variable
            i_IRFs = IRFs_to_plot[var]

            if ndims(i_IRFs) == 3
                # Extract the raw dynamic path
                i_IRFs_raw = i_IRFs[:, 1:horizon, idx]
                
                # Extract t=1 as the steady state (using 1:1 keeps it as a 2D matrix for broadcasting)
                ss_dist = i_IRFs_raw[:, 1:1]
                
                # Compute deviation from steady state
                i_IRFs_dev = i_IRFs_raw .- ss_dist

                p = plot_univariate_plot(
                    i_IRFs_dev,
                    var,
                    "Δ " * lab, # Updates title to indicate deviation
                    i_shock_lab,
                    horizon,
                    bounds,
                    n_par;
                    legend = false,
                )

                # Save plot
                if save_fig
                    savefig(
                        p,
                        path * "/DistIRFsDev_" * string(i_shock) * "_" * var * suffix * ".pdf",
                    )
                end

                # Show plot
                if show_fig
                    display(p)
                end

            elseif ndims(i_IRFs) == 4

                # Extract the raw dynamic path
                i_IRFs_raw = i_IRFs[:, :, 1:horizon, idx]
                
                # Extract t=1 as the steady state (using 1:1 keeps it as a 3D matrix for broadcasting)
                ss_dist = i_IRFs_raw[:, :, 1:1]
                
                # Compute deviation from steady state
                i_IRFs_dev = i_IRFs_raw .- ss_dist

                # Create plot
                anim = plot_bivariate_animation(
                    i_IRFs_dev,
                    var,
                    "Δ " * lab, # Updates title to indicate deviation
                    i_shock_lab,
                    horizon,
                    bounds,
                    n_par;
                    legend = legend,
                )

                # Save plot
                if save_fig
                    anim = gif(
                        anim,
                        path * "/DistIRFsDev_" * string(i_shock) * "_" * var * suffix * ".gif";
                        fps = fps,
                        show_msg = false,
                    )
                else
                    anim = gif(anim; fps = fps, show_msg = false)
                end

                # Show plot
                if show_fig
                    display(anim)
                end
            end
        end
    end
end