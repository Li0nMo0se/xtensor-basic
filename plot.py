import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse

# from https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, values[y], width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), bbox_to_anchor=(1.04,1), loc="upper left")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot benchmark.')
    parser.add_argument('--filename', type=str, help='Path to the csv filename', required=True)
    parser.add_argument('--log', action='store_true', help='Log scale on the y axis', default=False)
    parser.add_argument('--ymin', type=float, help='Minimum y axis', default=0.000001)
    parser.add_argument('--ymax', type=float, help='Maximum y axis')
    parser.add_argument("--rm_index", nargs="+", help='List of index to remove from the plot')

    # Parse
    args = parser.parse_args()

    # Check input file extension
    if args.filename.split('.')[-1] != "csv":
        raise ValueError(f"Expect a csv file. Got `{args.filename}``.")

    # Read input file csv
    df = pd.read_csv(args.filename, index_col=0)

    # Get output filenames
    output = ''.join(args.filename.split('.')[:-1]) + "_plot" + ("_log" if args.log else "") + ".png"
    output_bar = ''.join(args.filename.split('.')[:-1]) + "_bar" + ("_log" if args.log else "") + ".png"

    # Drop index
    if args.rm_index is not None:
        for index in args.rm_index:
            if int(index) in df.index:
                df.drop(int(index), axis=0, inplace=True)

    # If not ymax given (performed after dropping unrelevant index)
    if args.ymax is None:
        args.ymax = max(df.max(axis=1)) # Max all elements in the df


    # Plot lines plot
    plt.figure(figsize=(15,10), constrained_layout=True)
    for serie in df:
        plt.plot(df[serie], label=df[serie].name)

    plt.legend(title='Approaches', bbox_to_anchor=(1.04,1), loc="upper left")
    plt.title("Processing speed with the different approaches")
    plt.ylabel("time" + (" (log scale)" if args.log else ""), fontsize=14)
    plt.xlabel('ksize (log scale)', fontsize=14)
    plt.ylim(args.ymin, args.ymax)
    plt.xticks(df.index.values)
    if args.log:
        plt.xscale('log')
    plt.savefig(output)

    # Plot bar
    fig, ax = plt.subplots(figsize=(15,10), constrained_layout=True)
    if args.log:
        ax.set_yscale('log')
    bar_plot(ax, df.to_dict())
    plt.ylabel("time" + (" (log scale)" if args.log else ""), fontsize=14)
    plt.xlabel(f"{df.index.values}")
    plt.ylim(args.ymin, args.ymax)
    plt.title("Processing speed with the different approaches")
    plt.tick_params(labelbottom=False, bottom=False)
    plt.savefig(output_bar)


