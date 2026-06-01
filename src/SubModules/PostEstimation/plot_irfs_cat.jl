"""
    plot_irfs_cat(
        shock_categories,
        vars_to_plot,
        IRFs_to_plot,
        IRFs_order,
        ids;
        horizon = 40,
        factor = 100,
        fig_size = (1800, 0),
        ncols = 4,
        subplot_bottom_margin = nothing,
        subplot_left_margin = nothing,
        show_fig = true,
        save_fig = false,
        save_fig_indiv = false,
        path = "",
        suffix = "",
        yscale = "standard",
        style_options = (lw = 2, color = :auto, linestyle = :solid)
    )

Plots impulse response functions (IRFs) for specified shocks and variables, given IRFs as
produced by `compute_irfs`, organized by shock categories.

# Arguments

  - `shock_categories::Dict{Tuple{String,String},Vector{Symbol}}`: A dictionary where each
    key represents a category of shocks (with a label and a string for saving), and each
    value is a vector of symbols representing the shocks in that category.
  - `vars_to_plot::Vector{Tuple{Symbol,String}}`: A vector of tuples, each containing a
    variable to plot (as a `Symbol`) and its corresponding label (`String`).
  - `IRFs_to_plot::Array{Float64,3}`: A 3D array of IRFs.
  - `IRFs_order::Vector{Symbol}`: A vector of symbols specifying the order of shocks in the
    IRF arrays.
  - `ids`: A structure mapping variable symbols to their corresponding indices in the IRF
    arrays, which must be identical for all IRFs versions.

# Keyword Arguments

  - `horizon::Int64`: The time horizon (number of periods) over which IRFs are plotted.
    Default is `40`.
  - `factor::Int64`: Scaling factor for the IRFs (default: `100`).
  - `fig_size::Tuple{Int,Int}`: Total figure size `(width, height)`. When height is `0`
    (default), it is computed automatically as `250 * nrow + 150`.
  - `ncols::Int`: Number of subplot columns (default: `4`).
  - `subplot_bottom_margin`: Absolute padding below each subplot for x-axis tick labels.
    Pass a `Measures` value or leave as `nothing` to use `8 * Plots.mm`.
  - `subplot_left_margin`: Absolute padding left of each subplot for y-axis tick labels.
    Same convention as `subplot_bottom_margin` (default: `8 * Plots.mm`).
  - `show_fig::Bool`: If `true`, displays the plot. Default is `true`.
  - `save_fig::Bool`: If `true`, saves the combined plot as a PDF. Default is `false`.
  - `save_fig_indiv::Bool`: If `true`, saves individual plots for each variable or shock
    panel directly inside a category-specific folder. Default is `false`.
  - `path::String`: The directory path where the generated plots should be saved. Default is
    an empty string (no saving).
  - `suffix::String`: A suffix to append to the saved plot filenames. Default is an empty
    string.
  - `yscale::Union{String, Tuple{Number,Number}, Dict{Symbol,Tuple{Number,Number}}}`: Y-axis
    scaling specification. When set to `"common"`, computes a common y-axis limit across all
    subplots from the data; when provided as a tuple, uses it as `(ymin, ymax)`; when
    provided as a dictionary, applies specified y-axis limits for each variable. Default is
    `"standard"`, which applies default scaling.
  - `shock_labels::Dict{Symbol,String}`: Optional display names for shocks used in legend
    labels. Symbols absent from the dict fall back to `string(shock)`. Example:
    `Dict(:GI => "Gov. Investment", :Gshock => "Gov. Consumption")`.
  - `style_options::NamedTuple`: A named tuple specifying stylistic options for the plots,
    including line width (`lw`), color (default: `:auto`), and linestyle (default:
    `:solid`). Default is `(lw = 2, color = :auto, linestyle = :solid)`.
"""
function plot_irfs_cat(
    shock_categories::Dict{Tuple{String,String},Vector{Symbol}},
    vars_to_plot::Vector{Tuple{Symbol,String}},
    IRFs_to_plot::Array{Float64,3},
    IRFs_order::Vector{Symbol},
    ids;
    horizon::Int64    = 40,
    factor::Int64     = 100,
    fig_size::Tuple{Int,Int} = (1800, 0),
    ncols::Int        = 4,
    subplot_bottom_margin = nothing,
    subplot_left_margin   = nothing,
    shock_labels::Dict{Symbol,String} = Dict{Symbol,String}(),
    show_fig::Bool    = true,
    save_fig::Bool    = false,
    save_fig_indiv::Bool = false,
    path::String      = "",
    suffix::String    = "",
    yscale::Union{String,Tuple{Number,Number},Dict{Symbol,Tuple{Number,Number}}} = "standard",
    style_options::NamedTuple = (lw = 2, color = :auto, linestyle = :solid),
)
    # Layout: +1 for the shock-panel prepended to the variable panels
    n_subplots = length(vars_to_plot) + 1
    ncol       = ncols
    nrow       = ceil(Int, n_subplots / ncol)
    fig_height = fig_size[2] == 0 ? 250 * nrow + 150 : fig_size[2]
    fig_width  = fig_size[1]

    _bottom_margin = isnothing(subplot_bottom_margin) ? 8 * Plots.mm : subplot_bottom_margin
    _left_margin   = isnothing(subplot_left_margin)   ? 8 * Plots.mm : subplot_left_margin

    # General stylistic choices for the plots
    pp_layout = (
        dpi    = 600,
        size   = (fig_width, fig_height),
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        tickfont  = font(14, "Computer Modern"),
        titlefont = font(16, "Computer Modern"),
        legendfont = font(14, "Computer Modern"),
        bottom_margin = _bottom_margin,
        left_margin   = _left_margin,
        lw = style_options.lw,
    )

    # Unpack variables and labels
    vars = [vars_to_plot[i][1] for i in eachindex(vars_to_plot)]
    labs = [vars_to_plot[i][2] for i in eachindex(vars_to_plot)]

    # Define the base directory for the IRFs folder
    irfs_path = joinpath(path)
    mkpath(irfs_path)

    # Loop over categories
    for (category_name, category_shocks) in shock_categories

        # Find position of current shocks (category_shocks) in IRFs array (IRFs_order)
        idx = [findfirst(x -> x == i_shock, IRFs_order) for i_shock in category_shocks]

        # If one of the shocks is not found, skip the category and print a warning
        if any(isnothing, idx)
            @warn "One or more shocks in category $category_name not found in IRFs_order, skipped category."
            continue
        end

        # Extract IRFs for these shocks, round to 10 digits, and multiply by factor
        i_IRFs = mapround(IRFs_to_plot[:, :, idx]; digits = 10) .* factor
        n_IRFs = size(i_IRFs, 3)

        effective_color = if style_options.color == :auto
            pcol = palette(:auto)
            length(pcol) < n_IRFs ? vcat(pcol, fill(pcol[1], n_IRFs - length(pcol))) :
            pcol[1:n_IRFs]
        elseif isa(style_options.color, AbstractVector)
            length(style_options.color) < n_IRFs ?
            vcat(
                style_options.color,
                fill(style_options.color[1], n_IRFs - length(style_options.color)),
            ) : style_options.color[1:n_IRFs]
        else
            fill(style_options.color, n_IRFs)
        end

        effective_linestyle = if isa(style_options.linestyle, AbstractVector)
            length(style_options.linestyle) < n_IRFs ?
            vcat(
                style_options.linestyle,
                fill(style_options.linestyle[1], n_IRFs - length(style_options.linestyle)),
            ) : style_options.linestyle[1:n_IRFs]
        else
            fill(style_options.linestyle, n_IRFs)
        end

        # Determine y-axis limits for "common" yscale, tuple, or dictionary
        if yscale == "common"
            ymin, ymax = extrema(
                vcat(
                    [
                        i_IRFs[getfield(ids, var), 1:horizon, :] for
                        var in vars if hasfield(typeof(ids), var)
                    ]...,
                ),
            )
        elseif yscale isa Tuple{Number,Number}
            ymin, ymax = yscale
        elseif yscale isa Dict
            ylimits_per_variable = yscale
        else
            ymin, ymax = nothing, nothing
        end

        # Create a plot for each variable
        pp = []
        for (var, lab) in zip(vars, labs)
            if hasfield(typeof(ids), var)
                var_idx = getfield(ids, var)
                p = plot(
                    i_IRFs[var_idx, 1:horizon, 1];
                    title = lab,
                    label = get(shock_labels, category_shocks[1], string(category_shocks[1])),
                    lw = style_options.lw,
                    color = effective_color[1],
                    linestyle = effective_linestyle[1],
                )
                for j = 2:n_IRFs
                    plot!(
                        p,
                        i_IRFs[var_idx, 1:horizon, j];
                        label = get(shock_labels, category_shocks[j], string(category_shocks[j])),
                        lw = style_options.lw,
                        color = effective_color[j],
                        linestyle = effective_linestyle[j],
                    )
                end
                if yscale isa Dict && haskey(yscale, var)
                    ylims!(p, ylimits_per_variable[var]...)
                elseif yscale == "common" || yscale isa Tuple{Number,Number}
                    ylims!(p, ymin, ymax)
                end
                p = plot!(p; legend = false)
            else
                @printf "Variable %s not found in ids\n" var
                p = plot()
            end
            push!(pp, p)
        end

        # Shock panel prepended — shows each shock's own IRF with legend
        p_shock = plot(; title = category_name[1], lw = style_options.lw)
        for (i, var) in enumerate(category_shocks)
            if hasfield(typeof(ids), var)
                var_idx = getfield(ids, var)
                plot!(
                    p_shock,
                    i_IRFs[var_idx, 1:horizon, i];
                    label = get(shock_labels, var, string(var)),
                    lw = style_options.lw,
                    color = effective_color[i],
                    linestyle = effective_linestyle[i],
                )
            else
                @printf "Variable %s not found in ids\n" var
            end
        end
        pp = [p_shock; pp...]

        # Combine all plots in a single figure
        fig = plot(pp...; layout = (nrow, ncol), pp_layout...)

        if save_fig
            savefig(fig, joinpath(irfs_path, "IRF_" * category_name[2] * suffix * ".pdf"))
        end

        if save_fig_indiv
            category_path = joinpath(irfs_path, category_name[2])
            mkpath(category_path)
            for (i, p) in enumerate(pp)
                p = plot!(p; legend = true, pp_layout...)
                var_name = i == 1 ? "ShockPanel" : string(vars[i - 1])
                savefig(
                    p,
                    joinpath(
                        category_path,
                        "IRF_" * category_name[2] * "_" * var_name * suffix * ".pdf",
                    ),
                )
            end
        end

        show_fig && display(fig)
    end
end
